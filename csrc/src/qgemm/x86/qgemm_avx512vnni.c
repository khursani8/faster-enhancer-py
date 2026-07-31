/*
 * AVX-512 VNNI int8 GEMM (Cascade Lake / Ice Lake / Sapphire Rapids, Zen 4).
 *
 * zmm vpdpbusd processes 16 N-lanes per instruction (2 NR=8 packed blocks
 * concatenated). The +128 trick mirrors the AVX-VNNI path. For the
 * leftover single NR=8 block in N (when N % 16 == 8), an inline ymm
 * dpbusd helper is used instead of depending on the qgemm_avxvnni TU
 * (keeps per-file flag isolation clean).
 *
 * Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavxvnni
 * -mavx2 -mfma per-file. The fp16-inout GRU entry points add F16C in
 * target attributes for VCVTPH2PS/VCVTPS2PH.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#define FE_QPOST_AVX512 1
#define FE_QPOST_AVX2   1   /* for fe_qg_sigmoid8 fallback */
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../qgemm_simd_post.inl"

/* zmm 16-lane path. 4-way wsum dependency-chain split — same principle as
 * the AVX-VNNI counterpart: a single chain hits a k4_groups·5 cyc latency
 * floor; the 4-way split brings chain depth to ~k/4 and unhides dispatch.
 * Only 4 acc added, trivial register pressure. Bit-id preserved (integer
 * add associative). */
static inline __m512i fe_avx512_wsum_pair(const int8_t *Bp_lo,
                                          const int8_t *Bp_hi,
                                          int k4_groups) {
    __m512i ws0 = _mm512_setzero_si512(), ws1 = _mm512_setzero_si512();
    __m512i ws2 = _mm512_setzero_si512(), ws3 = _mm512_setzero_si512();
    const __m512i v128 = _mm512_set1_epi8((char)0x80);
    int g = 0;
    for (; g + 3 < k4_groups; g += 4) {
        __m256i blo0 = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)(g + 0) * 32));
        __m256i bhi0 = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)(g + 0) * 32));
        __m512i b0 = _mm512_inserti64x4(_mm512_castsi256_si512(blo0), bhi0, 1);
        __m256i blo1 = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)(g + 1) * 32));
        __m256i bhi1 = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)(g + 1) * 32));
        __m512i b1 = _mm512_inserti64x4(_mm512_castsi256_si512(blo1), bhi1, 1);
        __m256i blo2 = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)(g + 2) * 32));
        __m256i bhi2 = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)(g + 2) * 32));
        __m512i b2 = _mm512_inserti64x4(_mm512_castsi256_si512(blo2), bhi2, 1);
        __m256i blo3 = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)(g + 3) * 32));
        __m256i bhi3 = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)(g + 3) * 32));
        __m512i b3 = _mm512_inserti64x4(_mm512_castsi256_si512(blo3), bhi3, 1);
        ws0 = _mm512_dpbusd_epi32(ws0, v128, b0);
        ws1 = _mm512_dpbusd_epi32(ws1, v128, b1);
        ws2 = _mm512_dpbusd_epi32(ws2, v128, b2);
        ws3 = _mm512_dpbusd_epi32(ws3, v128, b3);
    }
    for (; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        ws0 = _mm512_dpbusd_epi32(ws0, v128, b);
    }
    return _mm512_add_epi32(_mm512_add_epi32(ws0, ws1),
                            _mm512_add_epi32(ws2, ws3));
}

/* One-pass +128 prelude over A (mirrors avxvnni). 64 B/iter via zmm. */
static inline void fe_avx512_prexor128_a(int8_t *A_q, int n_bytes) {
    const __m512i v128 = _mm512_set1_epi8((char)0x80);
    int i = 0;
    for (; i + 63 < n_bytes; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i *)(A_q + i));
        _mm512_storeu_si512((__m512i *)(A_q + i), _mm512_xor_si512(v, v128));
    }
    for (; i < n_bytes; ++i) A_q[i] = (int8_t)(A_q[i] ^ (int8_t)0x80);
}

/* Memory-source 32-bit broadcast for AVX-512 zmm — the zmm mirror of the
 * ymm-side fe_mem_broadcastd_epi32. GPR-routed _mm512_set1_epi32
 * lowers to `mov gpr,[mem]; vmovd xmm,gpr; vpbroadcastd zmm,xmm`, with
 * the broadcast on p5. vpdpbusd zmm dispatches on p0+p5 (0.5 IPC) on
 * Sapphire Rapids / Zen 4, so p5 contention from the A broadcast steals
 * dot-product slots. The explicit load+broadcast form below emits
 * `vpbroadcastd zmm,[mem]` (one µop on a load port), freeing p5. */
static inline __m512i fe_mem_broadcastd_epi32_512(const void *p) {
    return _mm512_broadcastd_epi32(_mm_loadu_si32(p));
}

/* Post-prelude: A is u8 in memory. No per-iter XOR. */
static inline __m512i fe_avx512_row_acc(int R, int g,
                                        const int8_t *A_q, int lda,
                                        __m512i b,
                                        __m512i acc) {
    __m512i a_u8 = fe_mem_broadcastd_epi32_512(A_q + R * lda + g * 4);
    return _mm512_dpbusd_epi32(acc, a_u8, b);
}

static inline void qgemm_kernel_8x16_avx512(int K,
                                            const int8_t *A_q, int lda,
                                            const int8_t *Bp_lo,
                                            const int8_t *Bp_hi,
                                            __m512i wsum,
                                            int32_t *C32, int ldc32) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm512_sub_epi32(c0, wsum);
    c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum);
    c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum);
    c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum);
    c7 = _mm512_sub_epi32(c7, wsum);

    _mm512_storeu_si512((__m512i *)(C32 + 0 * ldc32), c0);
    _mm512_storeu_si512((__m512i *)(C32 + 1 * ldc32), c1);
    _mm512_storeu_si512((__m512i *)(C32 + 2 * ldc32), c2);
    _mm512_storeu_si512((__m512i *)(C32 + 3 * ldc32), c3);
    _mm512_storeu_si512((__m512i *)(C32 + 4 * ldc32), c4);
    _mm512_storeu_si512((__m512i *)(C32 + 5 * ldc32), c5);
    _mm512_storeu_si512((__m512i *)(C32 + 6 * ldc32), c6);
    _mm512_storeu_si512((__m512i *)(C32 + 7 * ldc32), c7);
}

/* 12-row zmm tile: 12 accumulator chains saturate vpdpbusd zmm
 * throughput on 2-FMA-unit AVX-512 parts (SPR/Zen4), where 8 chains are
 * latency-bound. AVX-512 has 32 zmm regs so 12 acc + B + A-bcast fit with
 * room to spare. Bit-id: integer accumulation, tiling-invariant. */
static inline void qgemm_kernel_12x16_avx512(int K,
                                             const int8_t *A_q, int lda,
                                             const int8_t *Bp_lo,
                                             const int8_t *Bp_hi,
                                             __m512i wsum,
                                             int32_t *C32, int ldc32) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    __m512i c8 = _mm512_setzero_si512(), c9 = _mm512_setzero_si512();
    __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avx512_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avx512_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avx512_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avx512_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);
    c8 = _mm512_sub_epi32(c8, wsum); c9 = _mm512_sub_epi32(c9, wsum);
    c10 = _mm512_sub_epi32(c10, wsum); c11 = _mm512_sub_epi32(c11, wsum);
    _mm512_storeu_si512((__m512i *)(C32 + 0  * ldc32), c0);
    _mm512_storeu_si512((__m512i *)(C32 + 1  * ldc32), c1);
    _mm512_storeu_si512((__m512i *)(C32 + 2  * ldc32), c2);
    _mm512_storeu_si512((__m512i *)(C32 + 3  * ldc32), c3);
    _mm512_storeu_si512((__m512i *)(C32 + 4  * ldc32), c4);
    _mm512_storeu_si512((__m512i *)(C32 + 5  * ldc32), c5);
    _mm512_storeu_si512((__m512i *)(C32 + 6  * ldc32), c6);
    _mm512_storeu_si512((__m512i *)(C32 + 7  * ldc32), c7);
    _mm512_storeu_si512((__m512i *)(C32 + 8  * ldc32), c8);
    _mm512_storeu_si512((__m512i *)(C32 + 9  * ldc32), c9);
    _mm512_storeu_si512((__m512i *)(C32 + 10 * ldc32), c10);
    _mm512_storeu_si512((__m512i *)(C32 + 11 * ldc32), c11);
}

/* SIMD narrow tail (1..11 rows) for the zmm 16-lane path. Keeps the
 * prexor'd-u8 broadcast + wsum subtraction (a generic signed-int8 scalar kernel
 * would be wrong on the +128-shifted A, and slow). Used for the M%12 remainder
 * (Winograd NTiles=64 → after 8-aligned cap this is 0, but kept for safety). */
static inline void qgemm_tail_avx512_int32(int rows, int K,
                                           const int8_t *A_q, int lda,
                                           const int8_t *Bp_lo,
                                           const int8_t *Bp_hi,
                                           __m512i wsum,
                                           int32_t *C32, int ldc32) {
    int k4_groups = K / 4;
    for (int r = 0; r < rows; ++r) {
        __m512i c = _mm512_setzero_si512();
        for (int g = 0; g < k4_groups; ++g) {
            __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
            __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
            __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
            c = fe_avx512_row_acc(r, g, A_q, lda, b, c);
        }
        c = _mm512_sub_epi32(c, wsum);
        _mm512_storeu_si512((__m512i *)(C32 + (size_t)r * ldc32), c);
    }
}

/* ymm 8-lane leftover (single NR=8 block when N % 16 == 8). 4-way
 * wsum dependency-chain split — same principle as zmm version. */
static inline __m256i fe_avx512_ymm_wsum_block(const int8_t *Bp, int k4_groups) {
    __m256i ws0 = _mm256_setzero_si256(), ws1 = _mm256_setzero_si256();
    __m256i ws2 = _mm256_setzero_si256(), ws3 = _mm256_setzero_si256();
    const __m256i v128 = _mm256_set1_epi8((char)0x80);
    int g = 0;
    for (; g + 3 < k4_groups; g += 4) {
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 0) * 32));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 1) * 32));
        __m256i b2 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 2) * 32));
        __m256i b3 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 3) * 32));
        ws0 = _mm256_dpbusd_epi32(ws0, v128, b0);
        ws1 = _mm256_dpbusd_epi32(ws1, v128, b1);
        ws2 = _mm256_dpbusd_epi32(ws2, v128, b2);
        ws3 = _mm256_dpbusd_epi32(ws3, v128, b3);
    }
    for (; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        ws0 = _mm256_dpbusd_epi32(ws0, v128, b);
    }
    return _mm256_add_epi32(_mm256_add_epi32(ws0, ws1),
                            _mm256_add_epi32(ws2, ws3));
}

/* Memory-source 32-bit broadcast — same rationale as AVX2 / AVX-VNNI:
 * GPR-routed _mm256_set1_epi32 lowers to a p5-only vpbroadcastd ymm,xmm,
 * stealing slots from vpdpbusd (which also dispatches on p0+p5). The
 * explicit load+broadcast form below emits vpbroadcastd ymm,[mem] on the
 * load ports (p2/p3/pA), freeing p5 for the dot product. */
static inline __m256i fe_mem_broadcastd_epi32(const void *p) {
    return _mm256_broadcastd_epi32(_mm_loadu_si32(p));
}

static inline __m256i fe_avx512_ymm_row_acc(int R, int g,
                                            const int8_t *A_q, int lda,
                                            __m256i b,
                                            __m256i acc) {
    __m256i a_u8 = fe_mem_broadcastd_epi32(A_q + R * lda + g * 4);
    return _mm256_dpbusd_epi32(acc, a_u8, b);
}

static inline void qgemm_kernel_8x8_avx512_ymm(int K,
                                               const int8_t *A_q, int lda,
                                               const int8_t *Bp,
                                               __m256i wsum,
                                               int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx512_ymm_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_ymm_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_ymm_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_ymm_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_ymm_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_ymm_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_ymm_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_ymm_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

void qgemm_avx512vnni_int32(int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32) {
    const int MR      = FE_QGEMM_MR;
    const int NR      = FE_QGEMM_NR;
    const int NR_PAIR = 2 * NR;
    int k4_groups = (K + 3) / 4;
    fe_avx512_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;

    for (; nr + NR_PAIR <= N; nr += NR_PAIR, bn += 2) {
        const int8_t *B_lo = Bp + (size_t)(bn + 0) * k4_groups * NR * 4;
        const int8_t *B_hi = Bp + (size_t)(bn + 1) * k4_groups * NR * 4;
        __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile over 8-aligned remainder */
            qgemm_kernel_12x16_avx512(K, A_q + (size_t)mr * K, K,
                                      B_lo, B_hi, wsum,
                                      C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x16_avx512(K, A_q + (size_t)mr * K, K,
                                     B_lo, B_hi, wsum,
                                     C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) {
            qgemm_tail_avx512_int32(M - mr, K, A_q + (size_t)mr * K, K, B_lo, B_hi,
                                    wsum, C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
    }
    if (nr + NR <= N) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avx512_ymm_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x8_avx512_ymm(K, A_q + (size_t)mr * K, K,
                                        B_block, wsum,
                                        C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
        nr += NR; bn += 1;
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/*
 * fp32-out fused: zmm store at NR_eff=16 with ymm fallback for the
 * single NR=8 leftover. Op order: c32 -> fp32 -> fmadd(bias, scale*c32)
 * -> optional SiLU.
 */
static inline void fe_avx512_fused_store_row_fp32(__m512i acc, __m512 vs,
                                                  const float *bias_n0,
                                                  int act_silu,
                                                  float *C) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    __m512 vv = bias_n0
        ? _mm512_fmadd_ps(vc, vs, _mm512_loadu_ps(bias_n0))
        : _mm512_mul_ps(vc, vs);
    if (act_silu) vv = _mm512_mul_ps(vv, fe_qg_sigmoid16(vv));
    _mm512_storeu_ps(C, vv);
}

static inline void fe_avx512_ymm_fused_store_row_fp32(__m256i acc, __m256 vs,
                                                      const float *bias_n0,
                                                      int act_silu,
                                                      float *C) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 vv = bias_n0
        ? _mm256_fmadd_ps(vc, vs, _mm256_loadu_ps(bias_n0))
        : _mm256_mul_ps(vc, vs);
    if (act_silu) vv = _mm256_mul_ps(vv, fe_qg_sigmoid8(vv));
    _mm256_storeu_ps(C, vv);
}

static void
qgemm_kernel_8x16_avx512_fp32(int K,
                              const int8_t *A_q, int lda,
                              const int8_t *Bp_lo,
                              const int8_t *Bp_hi,
                              __m512i wsum,
                              const float *combined_scale_n0,
                              const float *bias_n0,
                              float *C, int ldc,
                              int act_silu) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);

    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avx512_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avx512_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avx512_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avx512_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avx512_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avx512_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avx512_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
}

static void
qgemm_kernel_12x16_avx512_fp32(int K,
                               const int8_t *A_q, int lda,
                               const int8_t *Bp_lo, const int8_t *Bp_hi,
                               __m512i wsum,
                               const float *combined_scale_n0,
                               const float *bias_n0,
                               float *C, int ldc, int act_silu) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    __m512i c8 = _mm512_setzero_si512(), c9 = _mm512_setzero_si512();
    __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avx512_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avx512_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avx512_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avx512_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);
    c8 = _mm512_sub_epi32(c8, wsum); c9 = _mm512_sub_epi32(c9, wsum);
    c10 = _mm512_sub_epi32(c10, wsum); c11 = _mm512_sub_epi32(c11, wsum);
    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32(c0,  vs, bias_n0, act_silu, C + 0  * ldc);
    fe_avx512_fused_store_row_fp32(c1,  vs, bias_n0, act_silu, C + 1  * ldc);
    fe_avx512_fused_store_row_fp32(c2,  vs, bias_n0, act_silu, C + 2  * ldc);
    fe_avx512_fused_store_row_fp32(c3,  vs, bias_n0, act_silu, C + 3  * ldc);
    fe_avx512_fused_store_row_fp32(c4,  vs, bias_n0, act_silu, C + 4  * ldc);
    fe_avx512_fused_store_row_fp32(c5,  vs, bias_n0, act_silu, C + 5  * ldc);
    fe_avx512_fused_store_row_fp32(c6,  vs, bias_n0, act_silu, C + 6  * ldc);
    fe_avx512_fused_store_row_fp32(c7,  vs, bias_n0, act_silu, C + 7  * ldc);
    fe_avx512_fused_store_row_fp32(c8,  vs, bias_n0, act_silu, C + 8  * ldc);
    fe_avx512_fused_store_row_fp32(c9,  vs, bias_n0, act_silu, C + 9  * ldc);
    fe_avx512_fused_store_row_fp32(c10, vs, bias_n0, act_silu, C + 10 * ldc);
    fe_avx512_fused_store_row_fp32(c11, vs, bias_n0, act_silu, C + 11 * ldc);
}

static void
qgemm_kernel_8x8_avx512_ymm_fp32(int K,
                                 const int8_t *A_q, int lda,
                                 const int8_t *Bp,
                                 __m256i wsum,
                                 const float *combined_scale_n0,
                                 const float *bias_n0,
                                 float *C, int ldc,
                                 int act_silu) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx512_ymm_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_ymm_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_ymm_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_ymm_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_ymm_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_ymm_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_ymm_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_ymm_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx512_ymm_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avx512_ymm_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
}


void qgemm_avx512vnni_fp32_fused(int M, int N, int K,
                                 const int8_t *A_q, const int8_t *Bp,
                                 const float *combined_scale, const float *bias,
                                 float *C, int ldc, int act_silu,
                                 int32_t *c32_tail) {
    const int MR      = FE_QGEMM_MR;
    const int NR      = FE_QGEMM_NR;
    const int NR_PAIR = 2 * NR;
    int k4_groups = (K + 3) / 4;
    fe_avx512_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;

    for (; nr + NR_PAIR <= N; nr += NR_PAIR, bn += 2) {
        const int8_t *B_lo = Bp + (size_t)(bn + 0) * k4_groups * NR * 4;
        const int8_t *B_hi = Bp + (size_t)(bn + 1) * k4_groups * NR * 4;
        __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
        const float *cs_n = combined_scale + nr;
        const float *bias_n = bias ? bias + nr : NULL;
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile over 8-aligned remainder */
            qgemm_kernel_12x16_avx512_fp32(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x16_avx512_fp32(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr + NR <= N) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avx512_ymm_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x8_avx512_ymm_fp32(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
        nr += NR; bn += 1;
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32 acc */
static inline void fe_avx512_fused_store_row_fp32_acc(__m512i acc, __m512 vs,
                                                      const float *bias_n0,
                                                      float *C) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    __m512 vv = bias_n0
        ? _mm512_fmadd_ps(vc, vs, _mm512_loadu_ps(bias_n0))
        : _mm512_mul_ps(vc, vs);
    vv = _mm512_add_ps(vv, _mm512_loadu_ps(C));
    _mm512_storeu_ps(C, vv);
}

static inline void fe_avx512_ymm_fused_store_row_fp32_acc(__m256i acc, __m256 vs,
                                                          const float *bias_n0,
                                                          float *C) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 vv = bias_n0
        ? _mm256_fmadd_ps(vc, vs, _mm256_loadu_ps(bias_n0))
        : _mm256_mul_ps(vc, vs);
    vv = _mm256_add_ps(vv, _mm256_loadu_ps(C));
    _mm256_storeu_ps(C, vv);
}

static void
qgemm_kernel_8x16_avx512_fp32_acc(int K,
                                  const int8_t *A_q, int lda,
                                  const int8_t *Bp_lo,
                                  const int8_t *Bp_hi,
                                  __m512i wsum,
                                  const float *combined_scale_n0,
                                  const float *bias_n0,
                                  float *C, int ldc) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);

    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
}

static void
qgemm_kernel_12x16_avx512_fp32_acc(int K,
                                   const int8_t *A_q, int lda,
                                   const int8_t *Bp_lo, const int8_t *Bp_hi,
                                   __m512i wsum,
                                   const float *combined_scale_n0,
                                   const float *bias_n0,
                                   float *C, int ldc) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    __m512i c8 = _mm512_setzero_si512(), c9 = _mm512_setzero_si512();
    __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avx512_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avx512_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avx512_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avx512_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);
    c8 = _mm512_sub_epi32(c8, wsum); c9 = _mm512_sub_epi32(c9, wsum);
    c10 = _mm512_sub_epi32(c10, wsum); c11 = _mm512_sub_epi32(c11, wsum);
    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32_acc(c0,  vs, bias_n0, C + 0  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c1,  vs, bias_n0, C + 1  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c2,  vs, bias_n0, C + 2  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c3,  vs, bias_n0, C + 3  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c4,  vs, bias_n0, C + 4  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c5,  vs, bias_n0, C + 5  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c6,  vs, bias_n0, C + 6  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c7,  vs, bias_n0, C + 7  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c8,  vs, bias_n0, C + 8  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c9,  vs, bias_n0, C + 9  * ldc);
    fe_avx512_fused_store_row_fp32_acc(c10, vs, bias_n0, C + 10 * ldc);
    fe_avx512_fused_store_row_fp32_acc(c11, vs, bias_n0, C + 11 * ldc);
}

static void
qgemm_kernel_8x8_avx512_ymm_fp32_acc(int K,
                                     const int8_t *A_q, int lda,
                                     const int8_t *Bp,
                                     __m256i wsum,
                                     const float *combined_scale_n0,
                                     const float *bias_n0,
                                     float *C, int ldc) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx512_ymm_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_ymm_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_ymm_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_ymm_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_ymm_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_ymm_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_ymm_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_ymm_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx512_ymm_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avx512_ymm_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
}


__attribute__((hot))
void qgemm_avx512vnni_fp32_fused_acc(int M, int N, int K,
                                     const int8_t *A_q, const int8_t *Bp,
                                     const float *combined_scale, const float *bias,
                                     float *C, int ldc, int32_t *c32_tail) {
    const int MR      = FE_QGEMM_MR;
    const int NR      = FE_QGEMM_NR;
    const int NR_PAIR = 2 * NR;
    int k4_groups = (K + 3) / 4;
    fe_avx512_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;

    for (; nr + NR_PAIR <= N; nr += NR_PAIR, bn += 2) {
        const int8_t *B_lo = Bp + (size_t)(bn + 0) * k4_groups * NR * 4;
        const int8_t *B_hi = Bp + (size_t)(bn + 1) * k4_groups * NR * 4;
        __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
        const float *cs_n = combined_scale + nr;
        const float *bias_n = bias ? bias + nr : NULL;
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile over 8-aligned remainder */
            qgemm_kernel_12x16_avx512_fp32_acc(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
        }
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x16_avx512_fp32_acc(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr + NR <= N) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avx512_ymm_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x8_avx512_ymm_fp32_acc(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
        nr += NR; bn += 1;
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32 track_maxabs */
static inline void fe_avx512_fused_store_row_fp32_track(__m512i acc, __m512 vs,
                                                        const float *bias_n0,
                                                        float *C,
                                                        __m512 *vmax) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    __m512 vv = bias_n0
        ? _mm512_fmadd_ps(vc, vs, _mm512_loadu_ps(bias_n0))
        : _mm512_mul_ps(vc, vs);
    _mm512_storeu_ps(C, vv);
    const __m512i sm = _mm512_set1_epi32(0x7FFFFFFF);
    __m512 va = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(vv), sm));
    *vmax = _mm512_max_ps(*vmax, va);
}

/*
 * ymm-width track helper: zmm vmax is the canonical accumulator. To
 * avoid needing AVX-512DQ insertf32x8, broadcast the ymm |vv| across
 * both halves of a zmm via inserti64x4 (AVX-512F). The per-half max is
 * correct because the final reduction takes max across all 16 lanes.
 */
static inline void fe_avx512_ymm_fused_store_row_fp32_track(__m256i acc, __m256 vs,
                                                            const float *bias_n0,
                                                            float *C,
                                                            __m512 *vmax) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 vv = bias_n0
        ? _mm256_fmadd_ps(vc, vs, _mm256_loadu_ps(bias_n0))
        : _mm256_mul_ps(vc, vs);
    _mm256_storeu_ps(C, vv);
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 va = _mm256_and_ps(vv, signmask);
    /* Broadcast ymm |vv| to both halves of zmm via inserti64x4. */
    __m512i vai = _mm512_castsi256_si512(_mm256_castps_si256(va));
    vai = _mm512_inserti64x4(vai, _mm256_castps_si256(va), 1);
    __m512 vaz = _mm512_castsi512_ps(vai);
    *vmax = _mm512_max_ps(*vmax, vaz);
}

static void
qgemm_kernel_8x16_avx512_fp32_track(int K,
                                    const int8_t *A_q, int lda,
                                    const int8_t *Bp_lo,
                                    const int8_t *Bp_hi,
                                    __m512i wsum,
                                    const float *combined_scale_n0,
                                    const float *bias_n0,
                                    float *C, int ldc,
                                    __m512 *vmax) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);

    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
}

static void
qgemm_kernel_12x16_avx512_fp32_track(int K,
                                     const int8_t *A_q, int lda,
                                     const int8_t *Bp_lo, const int8_t *Bp_hi,
                                     __m512i wsum,
                                     const float *combined_scale_n0,
                                     const float *bias_n0,
                                     float *C, int ldc, __m512 *vmax) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    __m512i c8 = _mm512_setzero_si512(), c9 = _mm512_setzero_si512();
    __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)g * 32));
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)g * 32));
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);
        c0 = fe_avx512_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avx512_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avx512_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avx512_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avx512_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);
    c8 = _mm512_sub_epi32(c8, wsum); c9 = _mm512_sub_epi32(c9, wsum);
    c10 = _mm512_sub_epi32(c10, wsum); c11 = _mm512_sub_epi32(c11, wsum);
    __m512 vs = _mm512_loadu_ps(combined_scale_n0);
    fe_avx512_fused_store_row_fp32_track(c0,  vs, bias_n0, C + 0  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c1,  vs, bias_n0, C + 1  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c2,  vs, bias_n0, C + 2  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c3,  vs, bias_n0, C + 3  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c4,  vs, bias_n0, C + 4  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c5,  vs, bias_n0, C + 5  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c6,  vs, bias_n0, C + 6  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c7,  vs, bias_n0, C + 7  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c8,  vs, bias_n0, C + 8  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c9,  vs, bias_n0, C + 9  * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c10, vs, bias_n0, C + 10 * ldc, vmax);
    fe_avx512_fused_store_row_fp32_track(c11, vs, bias_n0, C + 11 * ldc, vmax);
}

static void
qgemm_kernel_8x8_avx512_ymm_fp32_track(int K,
                                       const int8_t *A_q, int lda,
                                       const int8_t *Bp,
                                       __m256i wsum,
                                       const float *combined_scale_n0,
                                       const float *bias_n0,
                                       float *C, int ldc,
                                       __m512 *vmax) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx512_ymm_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avx512_ymm_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avx512_ymm_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avx512_ymm_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avx512_ymm_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avx512_ymm_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avx512_ymm_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avx512_ymm_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx512_ymm_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avx512_ymm_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
}

__attribute__((hot))
void qgemm_avx512vnni_fp32_fused_track_maxabs(int M, int N, int K,
                                              const int8_t *A_q, const int8_t *Bp,
                                              const float *combined_scale,
                                              const float *bias,
                                              float *C, int ldc,
                                              int32_t *c32_tail,
                                              float *max_abs_out) {
    const int MR      = FE_QGEMM_MR;
    const int NR      = FE_QGEMM_NR;
    const int NR_PAIR = 2 * NR;
    int k4_groups = (K + 3) / 4;
    fe_avx512_prexor128_a((int8_t *)A_q, M * K);
    __m512 vmax = _mm512_setzero_ps();
    int nr = 0, bn = 0;

    for (; nr + NR_PAIR <= N; nr += NR_PAIR, bn += 2) {
        const int8_t *B_lo = Bp + (size_t)(bn + 0) * k4_groups * NR * 4;
        const int8_t *B_hi = Bp + (size_t)(bn + 1) * k4_groups * NR * 4;
        __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
        const float *cs_n = combined_scale + nr;
        const float *bias_n = bias ? bias + nr : NULL;
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile over 8-aligned remainder */
            qgemm_kernel_12x16_avx512_fp32_track(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, &vmax);
        }
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x16_avx512_fp32_track(
                K, A_q + (size_t)mr * K, K, B_lo, B_hi, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, &vmax);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr + NR <= N) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avx512_ymm_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x8_avx512_ymm_fp32_track(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, &vmax);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
        nr += NR; bn += 1;
    }
    if (nr < N) fe_qgemm_tail_unsupported();
    *max_abs_out = _mm512_reduce_max_ps(vmax);
}

/*
 * K=20 specialisation: zmm 8x16 path for NR_PAIR blocks plus ymm 8x8 for
 * the single NR=8 leftover. k4_groups=5 fully unrolled.
 */
__attribute__((always_inline))
static inline void qgemm_kernel_8x16_avx512_k20(const int8_t *A_q, int lda,
                                                const int8_t *Bp_lo,
                                                const int8_t *Bp_hi,
                                                __m512i wsum,
                                                int32_t *C32, int ldc32) {
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();

    #define AVX512_K20_GROUP(G) do {                                                 \
        __m256i b_lo = _mm256_loadu_si256((const __m256i *)(Bp_lo + (size_t)(G)*32));\
        __m256i b_hi = _mm256_loadu_si256((const __m256i *)(Bp_hi + (size_t)(G)*32));\
        __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo), b_hi, 1);       \
        c0 = fe_avx512_row_acc(0, (G), A_q, lda, b, c0);                       \
        c1 = fe_avx512_row_acc(1, (G), A_q, lda, b, c1);                       \
        c2 = fe_avx512_row_acc(2, (G), A_q, lda, b, c2);                       \
        c3 = fe_avx512_row_acc(3, (G), A_q, lda, b, c3);                       \
        c4 = fe_avx512_row_acc(4, (G), A_q, lda, b, c4);                       \
        c5 = fe_avx512_row_acc(5, (G), A_q, lda, b, c5);                       \
        c6 = fe_avx512_row_acc(6, (G), A_q, lda, b, c6);                       \
        c7 = fe_avx512_row_acc(7, (G), A_q, lda, b, c7);                       \
    } while (0)
    AVX512_K20_GROUP(0);
    AVX512_K20_GROUP(1);
    AVX512_K20_GROUP(2);
    AVX512_K20_GROUP(3);
    AVX512_K20_GROUP(4);
    #undef AVX512_K20_GROUP

    c0 = _mm512_sub_epi32(c0, wsum); c1 = _mm512_sub_epi32(c1, wsum);
    c2 = _mm512_sub_epi32(c2, wsum); c3 = _mm512_sub_epi32(c3, wsum);
    c4 = _mm512_sub_epi32(c4, wsum); c5 = _mm512_sub_epi32(c5, wsum);
    c6 = _mm512_sub_epi32(c6, wsum); c7 = _mm512_sub_epi32(c7, wsum);

    _mm512_storeu_si512((__m512i *)(C32 + 0 * ldc32), c0);
    _mm512_storeu_si512((__m512i *)(C32 + 1 * ldc32), c1);
    _mm512_storeu_si512((__m512i *)(C32 + 2 * ldc32), c2);
    _mm512_storeu_si512((__m512i *)(C32 + 3 * ldc32), c3);
    _mm512_storeu_si512((__m512i *)(C32 + 4 * ldc32), c4);
    _mm512_storeu_si512((__m512i *)(C32 + 5 * ldc32), c5);
    _mm512_storeu_si512((__m512i *)(C32 + 6 * ldc32), c6);
    _mm512_storeu_si512((__m512i *)(C32 + 7 * ldc32), c7);
}

__attribute__((always_inline))
static inline void qgemm_kernel_8x8_avx512_ymm_k20(const int8_t *A_q, int lda,
                                                    const int8_t *Bp,
                                                    __m256i wsum,
                                                    int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();

    #define AVX512_YMM_K20_GROUP(G) do {                                              \
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(G) * 32));    \
        c0 = fe_avx512_ymm_row_acc(0, (G), A_q, lda, b, c0);                    \
        c1 = fe_avx512_ymm_row_acc(1, (G), A_q, lda, b, c1);                    \
        c2 = fe_avx512_ymm_row_acc(2, (G), A_q, lda, b, c2);                    \
        c3 = fe_avx512_ymm_row_acc(3, (G), A_q, lda, b, c3);                    \
        c4 = fe_avx512_ymm_row_acc(4, (G), A_q, lda, b, c4);                    \
        c5 = fe_avx512_ymm_row_acc(5, (G), A_q, lda, b, c5);                    \
        c6 = fe_avx512_ymm_row_acc(6, (G), A_q, lda, b, c6);                    \
        c7 = fe_avx512_ymm_row_acc(7, (G), A_q, lda, b, c7);                    \
    } while (0)
    AVX512_YMM_K20_GROUP(0);
    AVX512_YMM_K20_GROUP(1);
    AVX512_YMM_K20_GROUP(2);
    AVX512_YMM_K20_GROUP(3);
    AVX512_YMM_K20_GROUP(4);
    #undef AVX512_YMM_K20_GROUP

    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

void qgemm_avx512vnni_int32_k20(int M, int N,
                                const int8_t *A_q, const int8_t *Bp,
                                int32_t *C32, int ldc32) {
    enum { K = 20, MR = FE_QGEMM_MR, NR = FE_QGEMM_NR, NR_PAIR = 2 * FE_QGEMM_NR };
    enum { k4_groups = (K + 3) / 4 };
    fe_avx512_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;
    for (; nr + NR_PAIR <= N; nr += NR_PAIR, bn += 2) {
        const int8_t *B_lo = Bp + (size_t)(bn + 0) * k4_groups * NR * 4;
        const int8_t *B_hi = Bp + (size_t)(bn + 1) * k4_groups * NR * 4;
        __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x16_avx512_k20(A_q + (size_t)mr * K, K,
                                          B_lo, B_hi, wsum,
                                          C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr + NR <= N) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avx512_ymm_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + MR <= M; mr += MR) {
            qgemm_kernel_8x8_avx512_ymm_k20(A_q + (size_t)mr * K, K,
                                             B_block, wsum,
                                             C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
        nr += NR; bn += 1;
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/*
 * GRU full-fused helpers. The kernel below
 * (qgemm_avx512vnni_gru_full_fused_fp16inout) runs the zmm 16-lane path
 * (AVX512_ZMM_TILE_LOCAL, _mm512_dpbusd_epi32) for the N-block pairs and
 * falls back to this ymm 8x8 path only for the single NR=8 leftover
 * (D=72 -> 4 zmm pairs + 1 ymm). Cross-tier byte bit-id with the
 * avx2 / avxvnni tiers is NOT achieved by avoiding zmm: the sigmoid/tanh
 * rcp approximation is reproduced exactly inside the zmm path via
 * fe_avx512_rcp_match256 (qgemm_simd_post.inl), which rebuilds the 12-bit
 * VRCPPS result instead of using the native rcp14.
 */
static inline void fe_avx512_ymm_rzgate_row(__m256i acc,
                                             __m256 vs, __m256 vb,
                                             const float *ih_row,
                                             const float *bsum_n,
                                             float *band, int n_off, int ld_band,
                                             int r_idx) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 hh = _mm256_fmadd_ps(vc, vs, vb);
    __m256 pre = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(ih_row + n_off), hh),
                               _mm256_loadu_ps(bsum_n));
    __m256 g = fe_qg_sigmoid8(pre);
    _mm256_storeu_ps(band + r_idx * ld_band + n_off, g);
}


#define AVX512_YMM_8X8_TILE_BODY(K, A_q, lda, Bp, wsum,                      \
                                  c0, c1, c2, c3, c4, c5, c6, c7)             \
    do {                                                                      \
        const __m256i _v128 = _mm256_set1_epi8((char)0x80);                  \
        c0 = c1 = c2 = c3 = _mm256_setzero_si256();                          \
        c4 = c5 = c6 = c7 = _mm256_setzero_si256();                          \
        int _k4g = (K) / 4;                                                  \
        for (int _g = 0; _g < _k4g; ++_g) {                                  \
            __m256i _b = _mm256_loadu_si256(                                  \
                (const __m256i *)((Bp) + (size_t)_g * 32));                   \
            c0 = fe_avx512_ymm_row_acc(0, _g, (A_q), (lda), _b, c0);  \
            c1 = fe_avx512_ymm_row_acc(1, _g, (A_q), (lda), _b, c1);  \
            c2 = fe_avx512_ymm_row_acc(2, _g, (A_q), (lda), _b, c2);  \
            c3 = fe_avx512_ymm_row_acc(3, _g, (A_q), (lda), _b, c3);  \
            c4 = fe_avx512_ymm_row_acc(4, _g, (A_q), (lda), _b, c4);  \
            c5 = fe_avx512_ymm_row_acc(5, _g, (A_q), (lda), _b, c5);  \
            c6 = fe_avx512_ymm_row_acc(6, _g, (A_q), (lda), _b, c6);  \
            c7 = fe_avx512_ymm_row_acc(7, _g, (A_q), (lda), _b, c7);  \
        }                                                                     \
        c0 = _mm256_sub_epi32(c0, (wsum));                                   \
        c1 = _mm256_sub_epi32(c1, (wsum));                                   \
        c2 = _mm256_sub_epi32(c2, (wsum));                                   \
        c3 = _mm256_sub_epi32(c3, (wsum));                                   \
        c4 = _mm256_sub_epi32(c4, (wsum));                                   \
        c5 = _mm256_sub_epi32(c5, (wsum));                                   \
        c6 = _mm256_sub_epi32(c6, (wsum));                                   \
        c7 = _mm256_sub_epi32(c7, (wsum));                                   \
    } while (0)


/*
 * Full-fused GRU: W_ih @ x, W_hh @ h, and gate update in one pass.
 *
 * Per MR=8 row block: run W_ih @ Xq for the three gate slices, in-register
 * dequant (vc*vs_ih + vb_ih) into an fp32 ih_tile[MR, 3D] scratch, then
 * run the r/z/n W_hh sweeps reading from ih_tile.
 */
static inline void fe_avx512_ymm_dq_row_to_ih(__m256i acc,
                                              __m256 vs, __m256 vb,
                                              float *ih_tile_row) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 ih = _mm256_fmadd_ps(vc, vs, vb);
    _mm256_storeu_ps(ih_tile_row, ih);
}


/* AVX-512 VNNI fp16-inout variant: gru_h stored fp16, ngate epilogue
 * dual-stores state back as fp16. ymm dual-store. */
__attribute__((target("avx2,f16c,fma")))
static inline void fe_avx512_ymm_ngate_row_fp16inout(__m256i acc,
                                                       __m256 vs, __m256 vb,
                                                       const float *ih_row,
                                                       const float *bn_i_n,
                                                       const float *bn_h_n,
                                                       const float *r_band,
                                                       const float *z_band,
                                                       int n_off, int ld_band,
                                                       uint16_t *h_row_fp16,
                                                       float *h_out_row,
                                                       int r_idx) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 hh = _mm256_fmadd_ps(vc, vs, vb);
    __m256 bi = _mm256_loadu_ps(bn_i_n);
    __m256 bh = _mm256_loadu_ps(bn_h_n);
    __m256 ihp = _mm256_add_ps(_mm256_loadu_ps(ih_row + n_off), bi);
    __m256 hhp = _mm256_add_ps(hh, bh);
    __m256 rb = _mm256_loadu_ps(r_band + r_idx * ld_band + n_off);
    __m256 np = _mm256_fmadd_ps(rb, hhp, ihp);
    __m256 n  = fe_qg_tanh8(np);
    __m256 zb = _mm256_loadu_ps(z_band + r_idx * ld_band + n_off);
    __m256 ho = _mm256_cvtph_ps(_mm_loadu_si128(
                    (const __m128i *)(h_row_fp16 + n_off)));
    __m256 hn = _mm256_fmadd_ps(zb, _mm256_sub_ps(ho, n), n);
    _mm256_storeu_ps(h_out_row + n_off, hn);
    _mm_storeu_si128((__m128i *)(h_row_fp16 + n_off),
                     _mm256_cvtps_ph(hn,
                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

/* zmm (16-lane) GRU gate epilogues — proper AVX-512 width for the GRU
 * (was running ymm 8-lane = AVX-VNNI rate, no zmm benefit). Each processes
 * a 16-col chunk (= 2 NR=8 blocks). Bit-id with the ymm path: every op is
 * per-element (cvt/fmadd/add) and sigmoid16/tanh16 == sigmoid8/tanh8 (the
 * zmm rcp is forced to the same 12-bit VRCPPS as the ymm path), so 16-wide
 * == two 8-wide, byte-for-byte. */
static inline void fe_avx512_zmm_dq_row_to_ih(__m512i acc, __m512 vs, __m512 vb,
                                              float *ih_tile_row) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    _mm512_storeu_ps(ih_tile_row, _mm512_fmadd_ps(vc, vs, vb));
}
static inline void fe_avx512_zmm_rzgate_row(__m512i acc, __m512 vs, __m512 vb,
                                            const float *ih_row, const float *bsum_n,
                                            float *band, int n_off, int ld_band,
                                            int r_idx) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    __m512 hh = _mm512_fmadd_ps(vc, vs, vb);
    __m512 pre = _mm512_add_ps(_mm512_add_ps(_mm512_loadu_ps(ih_row + n_off), hh),
                               _mm512_loadu_ps(bsum_n));
    _mm512_storeu_ps(band + r_idx * ld_band + n_off, fe_qg_sigmoid16(pre));
}
static inline void fe_avx512_zmm_ngate_row_fp16inout(__m512i acc, __m512 vs, __m512 vb,
                                                     const float *ih_row,
                                                     const float *bn_i_n,
                                                     const float *bn_h_n,
                                                     const float *r_band,
                                                     const float *z_band,
                                                     int n_off, int ld_band,
                                                     uint16_t *h_row_fp16,
                                                     float *h_out_row, int r_idx) {
    __m512 vc = _mm512_cvtepi32_ps(acc);
    __m512 hh = _mm512_fmadd_ps(vc, vs, vb);
    __m512 ihp = _mm512_add_ps(_mm512_loadu_ps(ih_row + n_off), _mm512_loadu_ps(bn_i_n));
    __m512 hhp = _mm512_add_ps(hh, _mm512_loadu_ps(bn_h_n));
    __m512 rb = _mm512_loadu_ps(r_band + r_idx * ld_band + n_off);
    __m512 n  = fe_qg_tanh16(_mm512_fmadd_ps(rb, hhp, ihp));
    __m512 zb = _mm512_loadu_ps(z_band + r_idx * ld_band + n_off);
    __m512 ho = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(h_row_fp16 + n_off)));
    __m512 hn = _mm512_fmadd_ps(zb, _mm512_sub_ps(ho, n), n);
    _mm512_storeu_ps(h_out_row + n_off, hn);
    _mm256_storeu_si256((__m256i *)(h_row_fp16 + n_off),
                        _mm512_cvtps_ph(hn, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

__attribute__((hot))
__attribute__((target("avx2,f16c,fma,avx512f,avx512bw,avx512vl,avx512vnni")))
void qgemm_avx512vnni_gru_full_fused_fp16inout(int M, int D,
                                                const int8_t *Xq, int ld_x,
                                                const int8_t *Hq, int ld_h,
                                                const int8_t *Wq_ih,
                                                const int8_t *Wq_hh,
                                                const float *combined_ih,
                                                const float *bias_eff_ih,
                                                const float *combined_hh,
                                                const float *bias_eff_hh,
                                                const float *br_sum,
                                                const float *bz_sum,
                                                const float *bn_i,
                                                const float *bn_h,
                                                uint16_t *h_inout_fp16, int ld_h_inout,
                                                float *h_out_scratch, int ld_h_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    const int K  = D;
    const int k4_groups = (K + 3) / 4;
    const size_t weight_n_block_stride = (size_t)k4_groups * NR * 4;
    const int N_blocks_per_gate = D / NR;
    const int ld_band = D;
    float *r_band = fe_gru_r_band;
    float *z_band = fe_gru_z_band;
    static float ih_tile[FE_QGEMM_MR * FE_QGEMM_MAX_GRU_D3];
    const int ld_ih_tile = 3 * D;

    fe_avx512_prexor128_a((int8_t *)Xq, M * K);
    fe_avx512_prexor128_a((int8_t *)Hq, M * K);

    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
    __m512i z0, z1, z2, z3, z4, z5, z6, z7;

    #define AVX512_YMM_TILE_LOCAL(K, A_q, lda, Bp, wsum,                       \
                                  c0, c1, c2, c3, c4, c5, c6, c7)               \
        do {                                                                    \
            c0 = c1 = c2 = c3 = _mm256_setzero_si256();                        \
            c4 = c5 = c6 = c7 = _mm256_setzero_si256();                        \
            int _k4g = (K) / 4;                                                \
            for (int _g = 0; _g < _k4g; ++_g) {                                \
                __m256i _b = _mm256_loadu_si256(                                \
                    (const __m256i *)((Bp) + (size_t)_g * 32));                 \
                c0 = fe_avx512_ymm_row_acc(0, _g, (A_q), (lda), _b, c0);       \
                c1 = fe_avx512_ymm_row_acc(1, _g, (A_q), (lda), _b, c1);       \
                c2 = fe_avx512_ymm_row_acc(2, _g, (A_q), (lda), _b, c2);       \
                c3 = fe_avx512_ymm_row_acc(3, _g, (A_q), (lda), _b, c3);       \
                c4 = fe_avx512_ymm_row_acc(4, _g, (A_q), (lda), _b, c4);       \
                c5 = fe_avx512_ymm_row_acc(5, _g, (A_q), (lda), _b, c5);       \
                c6 = fe_avx512_ymm_row_acc(6, _g, (A_q), (lda), _b, c6);       \
                c7 = fe_avx512_ymm_row_acc(7, _g, (A_q), (lda), _b, c7);       \
            }                                                                   \
            c0 = _mm256_sub_epi32(c0, (wsum));                                 \
            c1 = _mm256_sub_epi32(c1, (wsum));                                 \
            c2 = _mm256_sub_epi32(c2, (wsum));                                 \
            c3 = _mm256_sub_epi32(c3, (wsum));                                 \
            c4 = _mm256_sub_epi32(c4, (wsum));                                 \
            c5 = _mm256_sub_epi32(c5, (wsum));                                 \
            c6 = _mm256_sub_epi32(c6, (wsum));                                 \
            c7 = _mm256_sub_epi32(c7, (wsum));                                 \
        } while (0)

    /* zmm 16-lane GRU GEMM tile (2 NR=8 blocks → one zmm). */
    #define AVX512_ZMM_TILE_LOCAL(K, A_q, lda, Bp_lo, Bp_hi, wsum,            \
                                  z0, z1, z2, z3, z4, z5, z6, z7)              \
        do {                                                                   \
            z0 = z1 = z2 = z3 = _mm512_setzero_si512();                       \
            z4 = z5 = z6 = z7 = _mm512_setzero_si512();                       \
            int _k4g = (K) / 4;                                               \
            for (int _g = 0; _g < _k4g; ++_g) {                               \
                __m256i _bl = _mm256_loadu_si256(                              \
                    (const __m256i *)((Bp_lo) + (size_t)_g * 32));             \
                __m256i _bh = _mm256_loadu_si256(                              \
                    (const __m256i *)((Bp_hi) + (size_t)_g * 32));             \
                __m512i _b = _mm512_inserti64x4(                              \
                    _mm512_castsi256_si512(_bl), _bh, 1);                      \
                z0 = fe_avx512_row_acc(0, _g, (A_q), (lda), _b, z0);          \
                z1 = fe_avx512_row_acc(1, _g, (A_q), (lda), _b, z1);          \
                z2 = fe_avx512_row_acc(2, _g, (A_q), (lda), _b, z2);          \
                z3 = fe_avx512_row_acc(3, _g, (A_q), (lda), _b, z3);          \
                z4 = fe_avx512_row_acc(4, _g, (A_q), (lda), _b, z4);          \
                z5 = fe_avx512_row_acc(5, _g, (A_q), (lda), _b, z5);          \
                z6 = fe_avx512_row_acc(6, _g, (A_q), (lda), _b, z6);          \
                z7 = fe_avx512_row_acc(7, _g, (A_q), (lda), _b, z7);          \
            }                                                                  \
            z0 = _mm512_sub_epi32(z0, (wsum)); z1 = _mm512_sub_epi32(z1,(wsum)); \
            z2 = _mm512_sub_epi32(z2, (wsum)); z3 = _mm512_sub_epi32(z3,(wsum)); \
            z4 = _mm512_sub_epi32(z4, (wsum)); z5 = _mm512_sub_epi32(z5,(wsum)); \
            z6 = _mm512_sub_epi32(z6, (wsum)); z7 = _mm512_sub_epi32(z7,(wsum)); \
        } while (0)

    /* Process each gate as zmm 16-col pairs + one ymm 8-col leftover
     * (D=72 → 4 pairs + 1). ZDQ/ZRZ/ZNG = zmm 16-lane epilogues; YDQ/YRZ/YNG
     * = ymm 8-lane leftover. Both share ih_tile/bands (contiguous cols). */
    #define ZIH(zk,R) fe_avx512_zmm_dq_row_to_ih(zk, vs, vb, row + (R)*ld_ih_tile)
    #define YIH(ck,R) fe_avx512_ymm_dq_row_to_ih(ck, ys, yb, row + (R)*ld_ih_tile)

    for (int mr = 0; mr + MR <= M; mr += MR) {
        const int8_t *X_block          = Xq + (size_t)mr * ld_x;
        const int8_t *H_block          = Hq + (size_t)mr * ld_h;
        uint16_t     *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float        *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;

        /* ---- W_ih @ X → ih_tile (3 gate slices) ---- */
        for (int slice = 0; slice < 3; ++slice) {
            const int gate_off = slice * D;
            const int base_blk = slice * N_blocks_per_gate;
            int nb = 0;
            for (; nb + 2 <= N_blocks_per_gate; nb += 2) {
                const int8_t *B_lo = Wq_ih + (size_t)(base_blk + nb)     * weight_n_block_stride;
                const int8_t *B_hi = Wq_ih + (size_t)(base_blk + nb + 1) * weight_n_block_stride;
                int n_off = nb * NR;
                __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
                AVX512_ZMM_TILE_LOCAL(K, X_block, ld_x, B_lo, B_hi, wsum,
                                      z0, z1, z2, z3, z4, z5, z6, z7);
                __m512 vs = _mm512_loadu_ps(combined_ih + gate_off + n_off);
                __m512 vb = _mm512_loadu_ps(bias_eff_ih + gate_off + n_off);
                float *row = ih_tile + gate_off + n_off;
                ZIH(z0,0); ZIH(z1,1); ZIH(z2,2); ZIH(z3,3);
                ZIH(z4,4); ZIH(z5,5); ZIH(z6,6); ZIH(z7,7);
            }
            for (; nb < N_blocks_per_gate; ++nb) {
                const int8_t *Bp = Wq_ih + (size_t)(base_blk + nb) * weight_n_block_stride;
                int n_off = nb * NR;
                __m256i wsum = fe_avx512_ymm_wsum_block(Bp, k4_groups);
                AVX512_YMM_TILE_LOCAL(K, X_block, ld_x, Bp, wsum,
                                      c0, c1, c2, c3, c4, c5, c6, c7);
                __m256 ys = _mm256_loadu_ps(combined_ih + gate_off + n_off);
                __m256 yb = _mm256_loadu_ps(bias_eff_ih + gate_off + n_off);
                float *row = ih_tile + gate_off + n_off;
                YIH(c0,0); YIH(c1,1); YIH(c2,2); YIH(c3,3);
                YIH(c4,4); YIH(c5,5); YIH(c6,6); YIH(c7,7);
            }
        }

        const float *ih_block = ih_tile;

        /* ---- r gate ---- */
        {
            int nb = 0;
            for (; nb + 2 <= N_blocks_per_gate; nb += 2) {
                const int8_t *B_lo = Wq_hh + (size_t)(nb)     * weight_n_block_stride;
                const int8_t *B_hi = Wq_hh + (size_t)(nb + 1) * weight_n_block_stride;
                int n_off = nb * NR;
                __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
                AVX512_ZMM_TILE_LOCAL(K, H_block, ld_h, B_lo, B_hi, wsum,
                                      z0, z1, z2, z3, z4, z5, z6, z7);
                __m512 vs = _mm512_loadu_ps(combined_hh + n_off);
                __m512 vb = _mm512_loadu_ps(bias_eff_hh + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m512i zk = (R==0?z0:R==1?z1:R==2?z2:R==3?z3:R==4?z4:R==5?z5:R==6?z6:z7);
                    fe_avx512_zmm_rzgate_row(zk, vs, vb, ih_block + R * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, R);
                }
            }
            for (; nb < N_blocks_per_gate; ++nb) {
                const int8_t *Bp = Wq_hh + (size_t)nb * weight_n_block_stride;
                int n_off = nb * NR;
                __m256i wsum = fe_avx512_ymm_wsum_block(Bp, k4_groups);
                AVX512_YMM_TILE_LOCAL(K, H_block, ld_h, Bp, wsum,
                                      c0, c1, c2, c3, c4, c5, c6, c7);
                __m256 ys = _mm256_loadu_ps(combined_hh + n_off);
                __m256 yb = _mm256_loadu_ps(bias_eff_hh + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m256i ck = (R==0?c0:R==1?c1:R==2?c2:R==3?c3:R==4?c4:R==5?c5:R==6?c6:c7);
                    fe_avx512_ymm_rzgate_row(ck, ys, yb, ih_block + R * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, R);
                }
            }
        }

        /* ---- z gate ---- */
        {
            const float *ih_z = ih_block + D;
            int nb = 0;
            for (; nb + 2 <= N_blocks_per_gate; nb += 2) {
                const int8_t *B_lo = Wq_hh + (size_t)(N_blocks_per_gate + nb)     * weight_n_block_stride;
                const int8_t *B_hi = Wq_hh + (size_t)(N_blocks_per_gate + nb + 1) * weight_n_block_stride;
                int n_off = nb * NR;
                __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
                AVX512_ZMM_TILE_LOCAL(K, H_block, ld_h, B_lo, B_hi, wsum,
                                      z0, z1, z2, z3, z4, z5, z6, z7);
                __m512 vs = _mm512_loadu_ps(combined_hh + D + n_off);
                __m512 vb = _mm512_loadu_ps(bias_eff_hh + D + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m512i zk = (R==0?z0:R==1?z1:R==2?z2:R==3?z3:R==4?z4:R==5?z5:R==6?z6:z7);
                    fe_avx512_zmm_rzgate_row(zk, vs, vb, ih_z + R * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, R);
                }
            }
            for (; nb < N_blocks_per_gate; ++nb) {
                const int8_t *Bp = Wq_hh + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
                int n_off = nb * NR;
                __m256i wsum = fe_avx512_ymm_wsum_block(Bp, k4_groups);
                AVX512_YMM_TILE_LOCAL(K, H_block, ld_h, Bp, wsum,
                                      c0, c1, c2, c3, c4, c5, c6, c7);
                __m256 ys = _mm256_loadu_ps(combined_hh + D + n_off);
                __m256 yb = _mm256_loadu_ps(bias_eff_hh + D + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m256i ck = (R==0?c0:R==1?c1:R==2?c2:R==3?c3:R==4?c4:R==5?c5:R==6?c6:c7);
                    fe_avx512_ymm_rzgate_row(ck, ys, yb, ih_z + R * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, R);
                }
            }
        }

        /* ---- n gate + h_new (fp16 dual-store) ---- */
        {
            const float *ih_n = ih_block + 2 * D;
            int nb = 0;
            for (; nb + 2 <= N_blocks_per_gate; nb += 2) {
                const int8_t *B_lo = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb)     * weight_n_block_stride;
                const int8_t *B_hi = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb + 1) * weight_n_block_stride;
                int n_off = nb * NR;
                __m512i wsum = fe_avx512_wsum_pair(B_lo, B_hi, k4_groups);
                AVX512_ZMM_TILE_LOCAL(K, H_block, ld_h, B_lo, B_hi, wsum,
                                      z0, z1, z2, z3, z4, z5, z6, z7);
                __m512 vs = _mm512_loadu_ps(combined_hh + 2 * D + n_off);
                __m512 vb = _mm512_loadu_ps(bias_eff_hh + 2 * D + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m512i zk = (R==0?z0:R==1?z1:R==2?z2:R==3?z3:R==4?z4:R==5?z5:R==6?z6:z7);
                    fe_avx512_zmm_ngate_row_fp16inout(zk, vs, vb, ih_n + R * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + R * ld_h_inout, h_block_scratch + R * ld_h_out, R);
                }
            }
            for (; nb < N_blocks_per_gate; ++nb) {
                const int8_t *Bp = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
                int n_off = nb * NR;
                __m256i wsum = fe_avx512_ymm_wsum_block(Bp, k4_groups);
                AVX512_YMM_TILE_LOCAL(K, H_block, ld_h, Bp, wsum,
                                      c0, c1, c2, c3, c4, c5, c6, c7);
                __m256 ys = _mm256_loadu_ps(combined_hh + 2 * D + n_off);
                __m256 yb = _mm256_loadu_ps(bias_eff_hh + 2 * D + n_off);
                for (int R = 0; R < 8; ++R) {
                    __m256i ck = (R==0?c0:R==1?c1:R==2?c2:R==3?c3:R==4?c4:R==5?c5:R==6?c6:c7);
                    fe_avx512_ymm_ngate_row_fp16inout(ck, ys, yb, ih_n + R * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + R * ld_h_inout, h_block_scratch + R * ld_h_out, R);
                }
            }
        }
    }
    #undef AVX512_YMM_TILE_LOCAL
    #undef AVX512_ZMM_TILE_LOCAL
    #undef ZIH
    #undef YIH
}

#undef AVX512_YMM_8X8_TILE_BODY

#endif /* x86 */

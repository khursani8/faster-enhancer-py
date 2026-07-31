/*
 * AVX-VNNI int8 GEMM (Alder Lake+, Zen 4+).
 *
 * vpdpbusd takes (u8, i8). We pre-shift the A buffer once per call
 * (XOR 0x80 over M*K bytes), then the inner loop is just broadcast +
 * vpdpbusd — no per-iteration vpxor. wsum subtract (128 * sum_K(b[n,k])
 * per output channel) absorbs the +128 shift.
 *
 * The previous design XORed each 4-byte A slice inside the inner loop,
 * which contended with vpbroadcastd and vpdpbusd on port 5 (oneDNN's
 * "s8/s8 is 0-15% slower than u8/s8" overhead). Moving the XOR to a
 * single vectorized prelude removes that hot-path cost; the prelude
 * itself is one pass over ~5-15 KB, trivially in L1.
 *
 * The pre-shift mutates A_q in place. Safe because every caller does a
 * fresh fe_quantize_activation immediately before invoking the kernel —
 * A is never reused across back-to-back kernel calls.
 *
 * Compiled with -mavx2 -mfma -mavxvnni per-file. The fp16-inout GRU
 * helpers add target("avx2,f16c,fma") / target("avx2,f16c,fma,avxvnni")
 * so VCVTPH2PS/VCVTPS2PH are emitted only inside the fp16 path.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#define FE_QPOST_AVX2 1
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../qgemm_simd_post.inl"

/*
 * AVX-VNNI exposes _mm256_dpbusd_avx_epi32; AVX-512VL exposes the same
 * encoding via _mm256_dpbusd_epi32. Use the AVX-only flavor so this TU
 * compiles with -mavxvnni alone.
 */
static inline __m256i fe_dpbusd256(__m256i acc, __m256i a_u8, __m256i b) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, a_u8, b);
#else
    return _mm256_dpbusd_avx_epi32(acc, a_u8, b);
#endif
}

/* wsum chain-split (4-way). Single-chain wsum has depth = k4_groups
 * vpdpbusds, all dependent on the previous acc → for K=72 (k4_groups=18)
 * latency floor = 18·5 = 90 cyc (latency-bound; dispatch is only 18 cyc).
 * 4-way split brings chain depth to ~5 → ~25 cyc latency, dispatch still
 * 18 cyc. Reduce: 3 vpaddd ≈ 1 cyc. Only 4 accumulators, so register
 * pressure is trivial (unlike the main 8x8 kernel where chain-split would
 * need 16 accs and spill).
 *
 * Bit-id: integer addition associative, vpdpbusd sat-free. Final result
 * identical to single-chain version. */
static inline __m256i fe_avxvnni_wsum_block(const int8_t *Bp, int k4_groups) {
    __m256i ws0 = _mm256_setzero_si256(), ws1 = _mm256_setzero_si256();
    __m256i ws2 = _mm256_setzero_si256(), ws3 = _mm256_setzero_si256();
    const __m256i v128 = _mm256_set1_epi8((char)0x80);
    int g = 0;
    for (; g + 3 < k4_groups; g += 4) {
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 0) * 32));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 1) * 32));
        __m256i b2 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 2) * 32));
        __m256i b3 = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(g + 3) * 32));
        ws0 = fe_dpbusd256(ws0, v128, b0);
        ws1 = fe_dpbusd256(ws1, v128, b1);
        ws2 = fe_dpbusd256(ws2, v128, b2);
        ws3 = fe_dpbusd256(ws3, v128, b3);
    }
    for (; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        ws0 = fe_dpbusd256(ws0, v128, b);
    }
    return _mm256_add_epi32(_mm256_add_epi32(ws0, ws1),
                            _mm256_add_epi32(ws2, ws3));
}

/* One-pass +128 prelude over A. Mutates A in place; caller commits A is
 * writable scratch (always true for aq_scratch). Vectorized at 32 B/iter. */
static inline void fe_avxvnni_prexor128_a(int8_t *A_q, int n_bytes) {
    const __m256i v128 = _mm256_set1_epi8((char)0x80);
    int i = 0;
    for (; i + 31 < n_bytes; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(A_q + i));
        _mm256_storeu_si256((__m256i *)(A_q + i), _mm256_xor_si256(v, v128));
    }
    for (; i < n_bytes; ++i) A_q[i] = (int8_t)(A_q[i] ^ (int8_t)0x80);
}

/* memory-source 32-bit broadcast. The natural-looking pattern
 *   int32_t a4 = *(const int32_t *)addr;
 *   __m256i v  = _mm256_set1_epi32(a4);
 * compiles to `mov gpr,[mem]; vmovd xmm,gpr; vpbroadcastd ymm,xmm` —
 * the YMM-from-XMM vpbroadcastd dispatches on **p5 only at 1.0 IPC**.
 * On Alder/Raptor Lake, vpdpbusd also dispatches dual on p0+p5 (0.5 IPC
 * total), so the row broadcasts steal half the vpdpbusd slots.
 *
 * Forcing the load + broadcast through explicit memory-source intrinsics
 * lets the compiler emit `vpbroadcastd ymm,[mem]` — a single uop on the
 * **load ports p2/p3/pA at 0.33 IPC**, leaving p5 free for vpdpbusd.
 *
 * Strict-aliasing-safe: _mm_loadu_si32 is the standard 32-bit unaligned
 * load intrinsic, treated by every major compiler as a memory-source
 * primitive (not a typed lvalue dereference). */
static inline __m256i fe_mem_broadcastd_epi32(const void *p) {
    return _mm256_broadcastd_epi32(_mm_loadu_si32(p));
}

/* Post-prelude: A is u8 in memory. Just broadcast + vpdpbusd, no XOR. */
static inline __m256i fe_avxvnni_row_acc(int R, int g,
                                         const int8_t *A_q, int lda,
                                         __m256i b,
                                         __m256i acc) {
    __m256i a_u8 = fe_mem_broadcastd_epi32(A_q + R * lda + g * 4);
    return fe_dpbusd256(acc, a_u8, b);
}

/* 12-row int32 tile (12 accumulator chains → vpdpbusd throughput
 * peak). k4_groups passed explicitly so the K=20 (k20) caller shares it. */
static inline void qgemm_kernel_12x8_avxvnni(int k4_groups,
                                             const int8_t *A_q, int lda,
                                             const int8_t *Bp,
                                             __m256i wsum,
                                             int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    __m256i c8 = _mm256_setzero_si256(), c9 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avxvnni_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avxvnni_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avxvnni_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avxvnni_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);
    c8 = _mm256_sub_epi32(c8, wsum); c9 = _mm256_sub_epi32(c9, wsum);
    c10 = _mm256_sub_epi32(c10, wsum); c11 = _mm256_sub_epi32(c11, wsum);
    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
    _mm256_storeu_si256((__m256i *)(C32 + 8 * ldc32), c8);
    _mm256_storeu_si256((__m256i *)(C32 + 9 * ldc32), c9);
    _mm256_storeu_si256((__m256i *)(C32 + 10 * ldc32), c10);
    _mm256_storeu_si256((__m256i *)(C32 + 11 * ldc32), c11);
}

/* Correct narrow M-tail (1..11 rows) for the prexor'd-u8 A buffer.
 * A generic signed-int8 scalar kernel (no wsum correction) would be WRONG on
 * the +128-shifted buffer — such a path was only ever safe because every M was
 * a multiple of MR=8 (no tail) before the 12-row tile. MR=12 introduces an
 * M%12 tail (e.g. Winograd NTiles=64 → 4 rows), so
 * the tail must keep the same broadcast + vpdpbusd + wsum path as the main
 * tile. */
static inline void qgemm_tail_avxvnni_int32(int rows, int k4_groups,
                                            const int8_t *A_q, int lda,
                                            const int8_t *Bp, __m256i wsum,
                                            int32_t *C32, int ldc32) {
    for (int r = 0; r < rows; ++r) {
        __m256i c = _mm256_setzero_si256();
        for (int g = 0; g < k4_groups; ++g) {
            __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
            c = fe_avxvnni_row_acc(r, g, A_q, lda, b, c);
        }
        c = _mm256_sub_epi32(c, wsum);
        _mm256_storeu_si256((__m256i *)(C32 + (size_t)r * ldc32), c);
    }
}

static inline void qgemm_kernel_8x8_avxvnni(int K,
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
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum);
    c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum);
    c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum);
    c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum);
    c7 = _mm256_sub_epi32(c7, wsum);

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

void qgemm_avxvnni_int32(int M, int N, int K,
                         const int8_t *A_q, const int8_t *Bp,
                         int32_t *C32, int ldc32) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    (void)MR;
    fe_avxvnni_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avxvnni_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile; leave 8-aligned remainder */
            qgemm_kernel_12x8_avxvnni(k4_groups, A_q + (size_t)mr * K, K,
                                      B_block, wsum,
                                      C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        for (; mr + 8 <= M; mr += 8) {
            qgemm_kernel_8x8_avxvnni(K, A_q + (size_t)mr * K, K,
                                     B_block, wsum,
                                     C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) {
            qgemm_tail_avxvnni_int32(M - mr, k4_groups, A_q + (size_t)mr * K, K,
                                     B_block, wsum,
                                     C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/*
 * fp32-out fused dequant + bias (+ optional SiLU). VNNI dot uses XOR-128
 * with wsum correction.
 */
static inline void fe_avxvnni_fused_store_row_fp32(__m256i acc, __m256 vs,
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

/*
 * 12-row tile. vpdpbusd on Golden/Raptor Cove is latency-5,
 * throughput-2/cyc (p0+p5). An 8-accumulator tile only sustains 8/5 =
 * 1.6 vpdpbusd/cyc (latency-bound, ~80% of peak); 12 independent
 * accumulator chains reach the full 2.0/cyc throughput ceiling. B stays
 * resident in one ymm per K-group; the 12 A-broadcasts are transient.
 * Reg budget: 12 acc + 1 B + 1 A-bcast = 14 ymm (no spill). Bit-id:
 * pure integer vpdpbusd accumulation, wider tiling reorders nothing
 * within any single output's dot product.
 */
static void
qgemm_kernel_12x8_avxvnni_fp32(const int8_t *A_q, int lda, int k4_groups,
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
    __m256i c8 = _mm256_setzero_si256(), c9 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avxvnni_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avxvnni_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avxvnni_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avxvnni_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);
    c8 = _mm256_sub_epi32(c8, wsum); c9 = _mm256_sub_epi32(c9, wsum);
    c10 = _mm256_sub_epi32(c10, wsum); c11 = _mm256_sub_epi32(c11, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avxvnni_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avxvnni_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avxvnni_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avxvnni_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avxvnni_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avxvnni_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avxvnni_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
    fe_avxvnni_fused_store_row_fp32(c8, vs, bias_n0, act_silu, C + 8 * ldc);
    fe_avxvnni_fused_store_row_fp32(c9, vs, bias_n0, act_silu, C + 9 * ldc);
    fe_avxvnni_fused_store_row_fp32(c10, vs, bias_n0, act_silu, C + 10 * ldc);
    fe_avxvnni_fused_store_row_fp32(c11, vs, bias_n0, act_silu, C + 11 * ldc);
}

static void
qgemm_kernel_8x8_avxvnni_fp32(int K,
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
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avxvnni_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avxvnni_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avxvnni_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avxvnni_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avxvnni_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avxvnni_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avxvnni_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
}


void qgemm_avxvnni_fp32_fused(int M, int N, int K,
                              const int8_t *A_q, const int8_t *Bp,
                              const float *combined_scale, const float *bias,
                              float *C, int ldc, int act_silu,
                              int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    (void)MR;
    fe_avxvnni_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avxvnni_wsum_block(B_block, k4_groups);
        const float *cs_n = combined_scale + nr;
        const float *bias_n = bias ? bias + nr : NULL;
        int mr = 0;
        for (; mr + 12 <= M; mr += 12) {
            qgemm_kernel_12x8_avxvnni_fp32(
                A_q + (size_t)mr * K, K, k4_groups, B_block, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        for (; mr + 8 <= M; mr += 8) {
            qgemm_kernel_8x8_avxvnni_fp32(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32-out acc */
static inline void fe_avxvnni_fused_store_row_fp32_acc(__m256i acc, __m256 vs,
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
qgemm_kernel_12x8_avxvnni_fp32_acc(const int8_t *A_q, int lda, int k4_groups,
                                   const int8_t *Bp,
                                   __m256i wsum,
                                   const float *combined_scale_n0,
                                   const float *bias_n0,
                                   float *C, int ldc) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    __m256i c8 = _mm256_setzero_si256(), c9 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avxvnni_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avxvnni_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avxvnni_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avxvnni_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);
    c8 = _mm256_sub_epi32(c8, wsum); c9 = _mm256_sub_epi32(c9, wsum);
    c10 = _mm256_sub_epi32(c10, wsum); c11 = _mm256_sub_epi32(c11, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c8, vs, bias_n0, C + 8 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c9, vs, bias_n0, C + 9 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c10, vs, bias_n0, C + 10 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c11, vs, bias_n0, C + 11 * ldc);
}

static void
qgemm_kernel_8x8_avxvnni_fp32_acc(int K,
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
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avxvnni_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
}


__attribute__((hot))
void qgemm_avxvnni_fp32_fused_acc(int M, int N, int K,
                                  const int8_t *A_q, const int8_t *Bp,
                                  const float *combined_scale, const float *bias,
                                  float *C, int ldc, int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    (void)MR;
    fe_avxvnni_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avxvnni_wsum_block(B_block, k4_groups);
        const float *cs_n = combined_scale + nr;
        const float *bias_n = bias ? bias + nr : NULL;
        int mr = 0;
        for (; mr + 12 <= M; mr += 12) {
            qgemm_kernel_12x8_avxvnni_fp32_acc(
                A_q + (size_t)mr * K, K, k4_groups, B_block, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
        }
        for (; mr + 8 <= M; mr += 8) {
            qgemm_kernel_8x8_avxvnni_fp32_acc(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32-out track_maxabs */
static inline void fe_avxvnni_fused_store_row_fp32_track(__m256i acc, __m256 vs,
                                                         const float *bias_n0,
                                                         float *C,
                                                         __m256 *vmax) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 vv = bias_n0
        ? _mm256_fmadd_ps(vc, vs, _mm256_loadu_ps(bias_n0))
        : _mm256_mul_ps(vc, vs);
    _mm256_storeu_ps(C, vv);
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    *vmax = _mm256_max_ps(*vmax, _mm256_and_ps(vv, signmask));
}

static void
qgemm_kernel_12x8_avxvnni_fp32_track(const int8_t *A_q, int lda, int k4_groups,
                                     const int8_t *Bp,
                                     __m256i wsum,
                                     const float *combined_scale_n0,
                                     const float *bias_n0,
                                     float *C, int ldc,
                                     __m256 *vmax) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    __m256i c8 = _mm256_setzero_si256(), c9 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
        c8 = fe_avxvnni_row_acc(8, g, A_q, lda, b, c8);
        c9 = fe_avxvnni_row_acc(9, g, A_q, lda, b, c9);
        c10 = fe_avxvnni_row_acc(10, g, A_q, lda, b, c10);
        c11 = fe_avxvnni_row_acc(11, g, A_q, lda, b, c11);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);
    c8 = _mm256_sub_epi32(c8, wsum); c9 = _mm256_sub_epi32(c9, wsum);
    c10 = _mm256_sub_epi32(c10, wsum); c11 = _mm256_sub_epi32(c11, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c8, vs, bias_n0, C + 8 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c9, vs, bias_n0, C + 9 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c10, vs, bias_n0, C + 10 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c11, vs, bias_n0, C + 11 * ldc, vmax);
}

static void
qgemm_kernel_8x8_avxvnni_fp32_track(int K,
                                    const int8_t *A_q, int lda,
                                    const int8_t *Bp,
                                    __m256i wsum,
                                    const float *combined_scale_n0,
                                    const float *bias_n0,
                                    float *C, int ldc,
                                    __m256 *vmax) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avxvnni_row_acc(0, g, A_q, lda, b, c0);
        c1 = fe_avxvnni_row_acc(1, g, A_q, lda, b, c1);
        c2 = fe_avxvnni_row_acc(2, g, A_q, lda, b, c2);
        c3 = fe_avxvnni_row_acc(3, g, A_q, lda, b, c3);
        c4 = fe_avxvnni_row_acc(4, g, A_q, lda, b, c4);
        c5 = fe_avxvnni_row_acc(5, g, A_q, lda, b, c5);
        c6 = fe_avxvnni_row_acc(6, g, A_q, lda, b, c6);
        c7 = fe_avxvnni_row_acc(7, g, A_q, lda, b, c7);
    }
    c0 = _mm256_sub_epi32(c0, wsum); c1 = _mm256_sub_epi32(c1, wsum);
    c2 = _mm256_sub_epi32(c2, wsum); c3 = _mm256_sub_epi32(c3, wsum);
    c4 = _mm256_sub_epi32(c4, wsum); c5 = _mm256_sub_epi32(c5, wsum);
    c6 = _mm256_sub_epi32(c6, wsum); c7 = _mm256_sub_epi32(c7, wsum);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avxvnni_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avxvnni_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
}

__attribute__((hot))
void qgemm_avxvnni_fp32_fused_track_maxabs(int M, int N, int K,
                                           const int8_t *A_q, const int8_t *Bp,
                                           const float *combined_scale,
                                           const float *bias,
                                           float *C, int ldc,
                                           int32_t *c32_tail,
                                           float *max_abs_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    fe_avxvnni_prexor128_a((int8_t *)A_q, M * K);
    (void)MR;
    __m256 vmax = _mm256_setzero_ps();
    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avxvnni_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + 12 <= M; mr += 12) {
            qgemm_kernel_12x8_avxvnni_fp32_track(
                A_q + (size_t)mr * K, K, k4_groups, B_block, wsum,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, &vmax);
        }
        for (; mr + 8 <= M; mr += 8) {
            qgemm_kernel_8x8_avxvnni_fp32_track(
                K, A_q + (size_t)mr * K, K, B_block, wsum,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, &vmax);
        }
        if (mr < M) fe_qgemm_tail_unsupported();
    }
    if (nr < N) fe_qgemm_tail_unsupported();
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 mx = _mm_max_ps(lo, hi);
    mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
    mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 0x55));
    _mm_store_ss(max_abs_out, mx);
}

/* K=20 specialisation: k4_groups=5 unrolled. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x8_avxvnni_k20(const int8_t *A_q, int lda,
                                                const int8_t *Bp,
                                                __m256i wsum,
                                                int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();

    #define AVXVNNI_K20_GROUP(G) do {                                                \
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)(G) * 32));   \
        c0 = fe_avxvnni_row_acc(0, (G), A_q, lda, b, c0);                      \
        c1 = fe_avxvnni_row_acc(1, (G), A_q, lda, b, c1);                      \
        c2 = fe_avxvnni_row_acc(2, (G), A_q, lda, b, c2);                      \
        c3 = fe_avxvnni_row_acc(3, (G), A_q, lda, b, c3);                      \
        c4 = fe_avxvnni_row_acc(4, (G), A_q, lda, b, c4);                      \
        c5 = fe_avxvnni_row_acc(5, (G), A_q, lda, b, c5);                      \
        c6 = fe_avxvnni_row_acc(6, (G), A_q, lda, b, c6);                      \
        c7 = fe_avxvnni_row_acc(7, (G), A_q, lda, b, c7);                      \
    } while (0)
    AVXVNNI_K20_GROUP(0);
    AVXVNNI_K20_GROUP(1);
    AVXVNNI_K20_GROUP(2);
    AVXVNNI_K20_GROUP(3);
    AVXVNNI_K20_GROUP(4);
    #undef AVXVNNI_K20_GROUP

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

void qgemm_avxvnni_int32_k20(int M, int N,
                             const int8_t *A_q, const int8_t *Bp,
                             int32_t *C32, int ldc32) {
    enum { K = 20, MR = FE_QGEMM_MR, NR = FE_QGEMM_NR };
    enum { k4_groups = (K + 3) / 4 };
    (void)MR;
    fe_avxvnni_prexor128_a((int8_t *)A_q, M * K);
    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        __m256i wsum = fe_avxvnni_wsum_block(B_block, k4_groups);
        int mr = 0;
        for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile; leave 8-aligned remainder */
            qgemm_kernel_12x8_avxvnni(k4_groups, A_q + (size_t)mr * K, K,
                                      B_block, wsum,
                                      C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        for (; mr + 8 <= M; mr += 8) {
            qgemm_kernel_8x8_avxvnni_k20(A_q + (size_t)mr * K, K,
                                          B_block, wsum,
                                          C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (mr < M) {
            qgemm_tail_avxvnni_int32(M - mr, k4_groups, A_q + (size_t)mr * K, K,
                                     B_block, wsum,
                                     C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/*
 * GRU full-fused helpers: VNNI XOR-128 + wsum per block, then dequant +
 * gate epilogue.
 */
static inline void fe_avxvnni_rzgate_row(__m256i acc,
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


/* fp16-in/inout-out variant. h_old loads via VCVTPH2PS, h_new writes
 * BOTH fp16 (storage update) and fp32 (scratch for rnn_fc). Dual store
 * eliminates both engine unpack and pack passes; the only added op vs
 * fp32-store is VCVTPS2PH (1 µop). */
__attribute__((target("avx2,f16c,fma")))
static inline void fe_avxvnni_ngate_row_fp16inout(__m256i acc,
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

#define AVXVNNI_8X8_TILE_BODY(K, A_q, lda, Bp, wsum,                         \
                              c0, c1, c2, c3, c4, c5, c6, c7)                 \
    do {                                                                      \
        c0 = c1 = c2 = c3 = _mm256_setzero_si256();                          \
        c4 = c5 = c6 = c7 = _mm256_setzero_si256();                          \
        int _k4g = (K) / 4;                                                  \
        for (int _g = 0; _g < _k4g; ++_g) {                                  \
            __m256i _b = _mm256_loadu_si256(                                  \
                (const __m256i *)((Bp) + (size_t)_g * 32));                   \
            c0 = fe_avxvnni_row_acc(0, _g, (A_q), (lda), _b, c0);     \
            c1 = fe_avxvnni_row_acc(1, _g, (A_q), (lda), _b, c1);     \
            c2 = fe_avxvnni_row_acc(2, _g, (A_q), (lda), _b, c2);     \
            c3 = fe_avxvnni_row_acc(3, _g, (A_q), (lda), _b, c3);     \
            c4 = fe_avxvnni_row_acc(4, _g, (A_q), (lda), _b, c4);     \
            c5 = fe_avxvnni_row_acc(5, _g, (A_q), (lda), _b, c5);     \
            c6 = fe_avxvnni_row_acc(6, _g, (A_q), (lda), _b, c6);     \
            c7 = fe_avxvnni_row_acc(7, _g, (A_q), (lda), _b, c7);     \
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

/* 12-row GRU tile: 12 vpdpbusd accumulator chains saturate the
 * lat-5/tput-2 unit (the 8-chain body topped out at 1.6/cyc). Mirrors the
 * non-GRU 12x8 kernels. M=freq is a multiple of 12 (72=6·12) on Medium, so
 * the GRU M-loop is exact (no tail). */
#define AVXVNNI_12_TILE_BODY(A_q, lda, Bp, k4g, wsum,                         \
                             c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11)           \
    do {                                                                      \
        c0=c1=c2=c3=c4=c5=_mm256_setzero_si256();                            \
        c6=c7=c8=c9=c10=c11=_mm256_setzero_si256();                          \
        for (int _g = 0; _g < (k4g); ++_g) {                                 \
            __m256i _b = _mm256_loadu_si256(                                  \
                (const __m256i *)((Bp) + (size_t)_g * 32));                   \
            c0 = fe_avxvnni_row_acc(0,  _g, (A_q), (lda), _b, c0);            \
            c1 = fe_avxvnni_row_acc(1,  _g, (A_q), (lda), _b, c1);            \
            c2 = fe_avxvnni_row_acc(2,  _g, (A_q), (lda), _b, c2);            \
            c3 = fe_avxvnni_row_acc(3,  _g, (A_q), (lda), _b, c3);            \
            c4 = fe_avxvnni_row_acc(4,  _g, (A_q), (lda), _b, c4);            \
            c5 = fe_avxvnni_row_acc(5,  _g, (A_q), (lda), _b, c5);            \
            c6 = fe_avxvnni_row_acc(6,  _g, (A_q), (lda), _b, c6);            \
            c7 = fe_avxvnni_row_acc(7,  _g, (A_q), (lda), _b, c7);            \
            c8 = fe_avxvnni_row_acc(8,  _g, (A_q), (lda), _b, c8);            \
            c9 = fe_avxvnni_row_acc(9,  _g, (A_q), (lda), _b, c9);            \
            c10= fe_avxvnni_row_acc(10, _g, (A_q), (lda), _b, c10);           \
            c11= fe_avxvnni_row_acc(11, _g, (A_q), (lda), _b, c11);           \
        }                                                                     \
        c0 =_mm256_sub_epi32(c0,(wsum)); c1 =_mm256_sub_epi32(c1,(wsum));     \
        c2 =_mm256_sub_epi32(c2,(wsum)); c3 =_mm256_sub_epi32(c3,(wsum));     \
        c4 =_mm256_sub_epi32(c4,(wsum)); c5 =_mm256_sub_epi32(c5,(wsum));     \
        c6 =_mm256_sub_epi32(c6,(wsum)); c7 =_mm256_sub_epi32(c7,(wsum));     \
        c8 =_mm256_sub_epi32(c8,(wsum)); c9 =_mm256_sub_epi32(c9,(wsum));     \
        c10=_mm256_sub_epi32(c10,(wsum));c11=_mm256_sub_epi32(c11,(wsum));    \
    } while (0)


/*
 * AVX-VNNI full-fused GRU: combines W_ih @ x, W_hh @ h, and the gate
 * update into a single row-block pass.
 */
static inline void fe_avxvnni_dq_row_to_ih(__m256i acc,
                                            __m256 vs, __m256 vb,
                                            float *ih_tile_row) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 ih = _mm256_fmadd_ps(vc, vs, vb);
    _mm256_storeu_ps(ih_tile_row, ih);
}


/*
 * fp16 inout variant. h_inout_fp16 is the fp16
 * recurrent state (read for h_old AND written with new state via
 * VCVTPS2PH). h_out_scratch is a fp32 scratch buffer that rnn_fc
 * consumes. Dual-store ngate_row eliminates BOTH engine unpack and
 * engine pack passes.
 */
__attribute__((hot))
__attribute__((target("avx2,f16c,fma,avxvnni")))
void qgemm_avxvnni_gru_full_fused_fp16inout(int M, int D,
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
    const int MR = FE_QGEMM_GRU_MR;   /* 12-row tile (vpdpbusd peak) */
    const int NR = FE_QGEMM_NR;
    const int K  = D;
    const int k4_groups = (K + 3) / 4;
    const size_t weight_n_block_stride = (size_t)k4_groups * NR * 4;
    const int N_blocks_per_gate = D / NR;
    const int ld_band = D;
    float *r_band = fe_gru_r_band;
    float *z_band = fe_gru_z_band;
    static float ih_tile[FE_QGEMM_GRU_MR * FE_QGEMM_MAX_GRU_D3];
    const int ld_ih_tile = 3 * D;

    fe_avxvnni_prexor128_a((int8_t *)Xq, M * K);
    fe_avxvnni_prexor128_a((int8_t *)Hq, M * K);

    /* wsum hoist: wsum = 128*sum_K(B) depends only on the (constant)
     * weights, not on the M-row block. Precompute all gate blocks once per
     * call instead of recomputing inside every M-tile (was MR-fold
     * redundant). 3 gates * N_blocks_per_gate each for ih and hh. */
    __m256i wsum_ih[3 * (FE_QGEMM_MAX_GRU_D / FE_QGEMM_NR)];
    __m256i wsum_hh[3 * (FE_QGEMM_MAX_GRU_D / FE_QGEMM_NR)];
    for (int blk = 0; blk < 3 * N_blocks_per_gate; ++blk) {
        wsum_ih[blk] = fe_avxvnni_wsum_block(
            Wq_ih + (size_t)blk * weight_n_block_stride, k4_groups);
        wsum_hh[blk] = fe_avxvnni_wsum_block(
            Wq_hh + (size_t)blk * weight_n_block_stride, k4_groups);
    }

    __m256i c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;

    for (int mr = 0; mr + MR <= M; mr += MR) {
        const int8_t *X_block          = Xq + (size_t)mr * ld_x;
        const int8_t *H_block          = Hq + (size_t)mr * ld_h;
        uint16_t     *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float        *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;

        /* W_ih @ Xq pass: dequant into ih_tile. */
        for (int slice = 0; slice < 3; ++slice) {
            const int gate_off = slice * D;
            for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
                const int8_t *Bp = Wq_ih +
                    (size_t)(slice * N_blocks_per_gate + nb) * weight_n_block_stride;
                int n_off = nb * NR;
                AVXVNNI_12_TILE_BODY(X_block, ld_x, Bp, k4_groups,
                                     wsum_ih[slice * N_blocks_per_gate + nb],
                                     c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
                __m256 vs = _mm256_loadu_ps(combined_ih + gate_off + n_off);
                __m256 vb = _mm256_loadu_ps(bias_eff_ih + gate_off + n_off);
                float *row = ih_tile + gate_off + n_off;
                fe_avxvnni_dq_row_to_ih(c0,  vs, vb, row + 0  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c1,  vs, vb, row + 1  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c2,  vs, vb, row + 2  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c3,  vs, vb, row + 3  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c4,  vs, vb, row + 4  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c5,  vs, vb, row + 5  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c6,  vs, vb, row + 6  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c7,  vs, vb, row + 7  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c8,  vs, vb, row + 8  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c9,  vs, vb, row + 9  * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c10, vs, vb, row + 10 * ld_ih_tile);
                fe_avxvnni_dq_row_to_ih(c11, vs, vb, row + 11 * ld_ih_tile);
            }
        }

        const float *ih_block = ih_tile;

        /* r gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp = Wq_hh + (size_t)nb * weight_n_block_stride;
            int n_off = nb * NR;
            AVXVNNI_12_TILE_BODY(H_block, ld_h, Bp, k4_groups, wsum_hh[nb],
                                 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + n_off);
            fe_avxvnni_rzgate_row(c0,  vs, vb, ih_block + 0  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 0);
            fe_avxvnni_rzgate_row(c1,  vs, vb, ih_block + 1  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 1);
            fe_avxvnni_rzgate_row(c2,  vs, vb, ih_block + 2  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 2);
            fe_avxvnni_rzgate_row(c3,  vs, vb, ih_block + 3  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 3);
            fe_avxvnni_rzgate_row(c4,  vs, vb, ih_block + 4  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 4);
            fe_avxvnni_rzgate_row(c5,  vs, vb, ih_block + 5  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 5);
            fe_avxvnni_rzgate_row(c6,  vs, vb, ih_block + 6  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 6);
            fe_avxvnni_rzgate_row(c7,  vs, vb, ih_block + 7  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 7);
            fe_avxvnni_rzgate_row(c8,  vs, vb, ih_block + 8  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 8);
            fe_avxvnni_rzgate_row(c9,  vs, vb, ih_block + 9  * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 9);
            fe_avxvnni_rzgate_row(c10, vs, vb, ih_block + 10 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 10);
            fe_avxvnni_rzgate_row(c11, vs, vb, ih_block + 11 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 11);
        }

        /* z gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp = Wq_hh +
                (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            AVXVNNI_12_TILE_BODY(H_block, ld_h, Bp, k4_groups,
                                 wsum_hh[N_blocks_per_gate + nb],
                                 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + D + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + D + n_off);
            const float *ih_z = ih_block + D;
            fe_avxvnni_rzgate_row(c0,  vs, vb, ih_z + 0  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 0);
            fe_avxvnni_rzgate_row(c1,  vs, vb, ih_z + 1  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 1);
            fe_avxvnni_rzgate_row(c2,  vs, vb, ih_z + 2  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 2);
            fe_avxvnni_rzgate_row(c3,  vs, vb, ih_z + 3  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 3);
            fe_avxvnni_rzgate_row(c4,  vs, vb, ih_z + 4  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 4);
            fe_avxvnni_rzgate_row(c5,  vs, vb, ih_z + 5  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 5);
            fe_avxvnni_rzgate_row(c6,  vs, vb, ih_z + 6  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 6);
            fe_avxvnni_rzgate_row(c7,  vs, vb, ih_z + 7  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 7);
            fe_avxvnni_rzgate_row(c8,  vs, vb, ih_z + 8  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 8);
            fe_avxvnni_rzgate_row(c9,  vs, vb, ih_z + 9  * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 9);
            fe_avxvnni_rzgate_row(c10, vs, vb, ih_z + 10 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 10);
            fe_avxvnni_rzgate_row(c11, vs, vb, ih_z + 11 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 11);
        }

        /* n gate + h_new — fp16 h_old read, fp32 h_new write. */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp = Wq_hh +
                (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            AVXVNNI_12_TILE_BODY(H_block, ld_h, Bp, k4_groups,
                                 wsum_hh[2 * N_blocks_per_gate + nb],
                                 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + 2 * D + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + 2 * D + n_off);
            const float *ih_n = ih_block + 2 * D;
            fe_avxvnni_ngate_row_fp16inout(c0,  vs, vb, ih_n + 0  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 0  * ld_h_inout, h_block_scratch + 0  * ld_h_out, 0);
            fe_avxvnni_ngate_row_fp16inout(c1,  vs, vb, ih_n + 1  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 1  * ld_h_inout, h_block_scratch + 1  * ld_h_out, 1);
            fe_avxvnni_ngate_row_fp16inout(c2,  vs, vb, ih_n + 2  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 2  * ld_h_inout, h_block_scratch + 2  * ld_h_out, 2);
            fe_avxvnni_ngate_row_fp16inout(c3,  vs, vb, ih_n + 3  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 3  * ld_h_inout, h_block_scratch + 3  * ld_h_out, 3);
            fe_avxvnni_ngate_row_fp16inout(c4,  vs, vb, ih_n + 4  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 4  * ld_h_inout, h_block_scratch + 4  * ld_h_out, 4);
            fe_avxvnni_ngate_row_fp16inout(c5,  vs, vb, ih_n + 5  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 5  * ld_h_inout, h_block_scratch + 5  * ld_h_out, 5);
            fe_avxvnni_ngate_row_fp16inout(c6,  vs, vb, ih_n + 6  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 6  * ld_h_inout, h_block_scratch + 6  * ld_h_out, 6);
            fe_avxvnni_ngate_row_fp16inout(c7,  vs, vb, ih_n + 7  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 7  * ld_h_inout, h_block_scratch + 7  * ld_h_out, 7);
            fe_avxvnni_ngate_row_fp16inout(c8,  vs, vb, ih_n + 8  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 8  * ld_h_inout, h_block_scratch + 8  * ld_h_out, 8);
            fe_avxvnni_ngate_row_fp16inout(c9,  vs, vb, ih_n + 9  * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 9  * ld_h_inout, h_block_scratch + 9  * ld_h_out, 9);
            fe_avxvnni_ngate_row_fp16inout(c10, vs, vb, ih_n + 10 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 10 * ld_h_inout, h_block_scratch + 10 * ld_h_out, 10);
            fe_avxvnni_ngate_row_fp16inout(c11, vs, vb, ih_n + 11 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 11 * ld_h_inout, h_block_scratch + 11 * ld_h_out, 11);
        }
    }
}

#undef AVXVNNI_8X8_TILE_BODY
#undef AVXVNNI_12_TILE_BODY

#endif /* x86 */

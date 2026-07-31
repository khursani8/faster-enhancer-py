/*
 * Haswell+ AVX2 int8 GEMM. Two dot paths, selected by qgemm_avx2_int32:
 * an i16 (vpmovsxbw+vpmaddwd, XNNPACK-style) PRIMARY for even-K / small-MK
 * (this model's even-K shapes), and a saturation-free vpsignb FALLBACK for
 * K-odd / large-MK. The vpsignb rationale is documented first (it was the
 * original choice); the i16 path that superseded it as default is below
 * (search "vpmovsxbw + vpmaddwd path").
 *
 * --- vpsignb fallback rationale ---
 * AVX2 has no native int8 dot product. The classic widen-via-vpmovsxbw +
 * vpmaddwd path needs vphaddd + vpermq to land per-lane sums, both of
 * which contend on port 5; VTune measured ~100% port-5 saturation. (The i16
 * primary below avoids this with a K-pair-interleaved repack instead of
 * vphaddd+vpermq.)
 *
 * The vpsignb trick (used by ggml/llama.cpp Q8_0) routes signed×signed
 * through unsigned×signed vpmaddubsw without saturation:
 *
 *   ax = vpsignb(a, a) = |a|        — u8 in [0, 127] (activation clamped)
 *   sy = vpsignb(b, a) = sign(a)·b  — i8 in [-127, +127] (weight clamped)
 *   p16 = vpmaddubsw(ax, sy)        — i16 pair sum, max |127·127·2| = 32258 < INT16_MAX
 *   p32 = vpmaddwd(p16, ones16)     — i32, sum-of-pair-sums = sum_K(a*b)
 *
 * The edge case `a=-128 AND b=-128` (psignb cannot negate -128) is
 * eliminated by the universal [-127, +127] clamp at quantize time:
 * weights in tools/quantize_bin.py + fe_qgemm_pack_W, activations in
 * qgemm_quant.c. The weight clamp is sufficient for the psignb edge case;
 * the activation clamp also keeps the i16 saturation budget identical to
 * the other signed×signed tiers.
 *
 * Port distribution shifts off the shuffle port: 2× vpsignb + 1×
 * vpmaddubsw + 1× vpmaddwd + 1× vpaddd per row group, all lane-internal,
 * no cross-lane vpermq.
 *
 * Compiled with -mavx2 -mfma per-file. The fp16-inout GRU entry points add
 * target("avx2,f16c,fma") so the TU-level flag list stays narrow while the
 * fp16 path still emits VCVTPH2PS/VCVTPS2PH.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#define FE_QPOST_AVX2 1
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../qgemm_simd_post.inl"

/* memory-source 32-bit broadcast. The natural-looking
 *   int32_t a4 = *(const int32_t *)addr;  __m256i v = _mm256_set1_epi32(a4);
 * pattern compiles to `mov gpr,[mem]; vmovd xmm,gpr; vpbroadcastd ymm,xmm`,
 * dispatching the broadcast on **p5 only at 1.0 IPC**. On Alder/Raptor Lake
 * vpdpbusd shares p0+p5; the broadcast steals half its slots. The explicit
 * load+broadcast form below lowers to `vpbroadcastd ymm,[mem]` on the load
 * ports (p2/p3/pA, 0.33 IPC), freeing p5. AVX2 vpsignb / vpmaddubsw /
 * vpmaddwd all run on p01 (no shuffle pressure), so this purely
 * dedupes the broadcast off the shared port. */
static inline __m256i fe_mem_broadcastd_epi32(const void *p) {
    return _mm256_broadcastd_epi32(_mm_loadu_si32(p));
}

/* vpmovsxbw + vpmaddwd path (XNNPACK-style). Drops the 16× vpsignb per
 * K-quartet (8 cyc on p15) by sign-extending A and B to i16 once and using
 * vpmaddwd directly. vpmaddwd is i16×i16 → i32 with adjacent-pair-add;
 * in faster-enhancer.c's int8 domain the worst pair sum is 2*127*127 = 32258, so bit-id
 * is preserved without saturation.
 *
 * Port shift: the prior vpsignb+vpmaddubsw chain had 32 p1 ops per K-quartet
 * (vpsignb 16 + vpmaddubsw 8 + vpmaddwd 8). This path has 8 vpmaddwd on p01 +
 * 1 B-side vpmovsxbw on p5. About 50% less p01 dispatch per K-quartet.
 *
 * Requires (a) A pre-expanded to i16 once per call, (b) B re-packed to
 * K-pair-interleaved layout per N-block. Both done in scratch buffers
 * below; total static cost ~195 KiB BSS. */
#define FE_AVX2_I16_MAX_MK   (256 * FE_QGEMM_MAX_K)   /* max M * K in any single call */
static FE_ALIGN64 int16_t fe_avx2_a_i16_buf[FE_AVX2_I16_MAX_MK];
static FE_ALIGN64 int8_t  fe_avx2_b_kpair_buf[FE_QGEMM_NR * FE_QGEMM_MAX_K];

/* Pre-expand A from i8 to i16 once per kernel call. Vectorized at 16 B/iter
 * via vpmovsxbw ymm,xmm (which dispatches as load+sxbw fused on Raptor Cove). */
static inline void fe_avx2_i16_expand_a(const int8_t *A_q, int M, int K, int16_t *out) {
    int total = M * K;
    int i = 0;
    for (; i + 15 < total; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(A_q + i));
        _mm256_storeu_si256((__m256i *)(out + i), _mm256_cvtepi8_epi16(a));
    }
    for (; i < total; ++i) {
        out[i] = (int16_t)A_q[i];
    }
}

/* Re-pack one N-block of B from the global K-quartet layout (col-major
 * within block: B[k=0..3, n=0..7]) to K-pair-interleaved (per-pair block
 * holds: B[k0,n0], B[k1,n0], B[k0,n1], B[k1,n1], ..., B[k0,n7], B[k1,n7]).
 *
 * After re-pack, vpmovsxbw of one 16-byte K-pair block produces 16 i16 with
 * the per-column pair (k0, k1) adjacent, which is exactly the layout
 * vpmaddwd's adjacent-pair-add operates on. Cost ~k4_groups·32 scalar byte
 * moves per N-block; <2% of inner-kernel cost. */
static inline void fe_avx2_i16_repack_b(const int8_t *Bp, int k4_groups, int8_t *out) {
    for (int kq = 0; kq < k4_groups; ++kq) {
        const int8_t *src = Bp + (size_t)kq * 32;
        int8_t *dst_k01 = out + (size_t)(kq * 2 + 0) * 16;
        int8_t *dst_k23 = out + (size_t)(kq * 2 + 1) * 16;
        for (int n = 0; n < 8; ++n) {
            dst_k01[n * 2 + 0] = src[n * 4 + 0];   /* col n K=0 */
            dst_k01[n * 2 + 1] = src[n * 4 + 1];   /* col n K=1 */
            dst_k23[n * 2 + 0] = src[n * 4 + 2];   /* col n K=2 */
            dst_k23[n * 2 + 1] = src[n * 4 + 3];   /* col n K=3 */
        }
    }
}

/* Per-row K-pair accumulator. a_bcast = (a_kp0_i16, a_kp1_i16) broadcast to
 * all 8 dword lanes; b_i16 = (col0 k0,k1, col1 k0,k1, ..., col7 k0,k1) from
 * the re-packed B. vpmaddwd → 8 i32, each = a_kp0·B[k0,n] + a_kp1·B[k1,n]. */
static inline __m256i fe_avx2_i16_row_acc(int r, int kp,
                                          const int16_t *A_i16, int lda_i16,
                                          __m256i b_i16, __m256i acc) {
    __m256i a_bcast = _mm256_broadcastd_epi32(
        _mm_loadu_si32(A_i16 + (size_t)r * lda_i16 + (size_t)kp * 2));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a_bcast, b_i16));
}

/* vpmovsxbw+vpmaddwd 8x8 tile body: produces 8 i32 accumulators (c0..c7). The caller
 * applies the appropriate epilogue (int32 store, fp32 dequant+bias+SiLU,
 * accumulate, track_maxabs). */
#define AVX2_I16_8X8_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,                  \
                                c0, c1, c2, c3, c4, c5, c6, c7)               \
    do {                                                                       \
        (c0) = (c1) = (c2) = (c3) = _mm256_setzero_si256();                   \
        (c4) = (c5) = (c6) = (c7) = _mm256_setzero_si256();                   \
        int _kp_count = (K) / 2;                                              \
        for (int _kp = 0; _kp < _kp_count; ++_kp) {                           \
            __m256i _b = _mm256_cvtepi8_epi16(                                \
                _mm_loadu_si128((const __m128i *)((Bp_kpair) + (size_t)_kp * 16))); \
            (c0) = fe_avx2_i16_row_acc(0, _kp, (A_i16), (lda_i16), _b, (c0)); \
            (c1) = fe_avx2_i16_row_acc(1, _kp, (A_i16), (lda_i16), _b, (c1)); \
            (c2) = fe_avx2_i16_row_acc(2, _kp, (A_i16), (lda_i16), _b, (c2)); \
            (c3) = fe_avx2_i16_row_acc(3, _kp, (A_i16), (lda_i16), _b, (c3)); \
            (c4) = fe_avx2_i16_row_acc(4, _kp, (A_i16), (lda_i16), _b, (c4)); \
            (c5) = fe_avx2_i16_row_acc(5, _kp, (A_i16), (lda_i16), _b, (c5)); \
            (c6) = fe_avx2_i16_row_acc(6, _kp, (A_i16), (lda_i16), _b, (c6)); \
            (c7) = fe_avx2_i16_row_acc(7, _kp, (A_i16), (lda_i16), _b, (c7)); \
        }                                                                      \
    } while (0)

/* 12-row vpmovsxbw+vpmaddwd tile body: 12 i32 accumulators. The inner op is
 * vpmaddwd + vpaddd; the vpaddd accumulate-chain has latency 1, so 8 rows
 * already hide it (this path is throughput-bound, not latency-bound like
 * vpdpbusd). 12 rows still shaves a little by amortising the B-load and the
 * per-K-pair sxbw across more accumulators. Bit-id: integer accumulation. */
#define AVX2_I16_12_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,                   \
                               c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11)        \
    do {                                                                      \
        (c0)=(c1)=(c2)=(c3)=(c4)=(c5)=_mm256_setzero_si256();                \
        (c6)=(c7)=(c8)=(c9)=(c10)=(c11)=_mm256_setzero_si256();              \
        int _kp_count = (K) / 2;                                             \
        for (int _kp = 0; _kp < _kp_count; ++_kp) {                          \
            __m256i _b = _mm256_cvtepi8_epi16(                               \
                _mm_loadu_si128((const __m128i *)((Bp_kpair) + (size_t)_kp * 16))); \
            (c0) = fe_avx2_i16_row_acc(0,  _kp, (A_i16), (lda_i16), _b, (c0)); \
            (c1) = fe_avx2_i16_row_acc(1,  _kp, (A_i16), (lda_i16), _b, (c1)); \
            (c2) = fe_avx2_i16_row_acc(2,  _kp, (A_i16), (lda_i16), _b, (c2)); \
            (c3) = fe_avx2_i16_row_acc(3,  _kp, (A_i16), (lda_i16), _b, (c3)); \
            (c4) = fe_avx2_i16_row_acc(4,  _kp, (A_i16), (lda_i16), _b, (c4)); \
            (c5) = fe_avx2_i16_row_acc(5,  _kp, (A_i16), (lda_i16), _b, (c5)); \
            (c6) = fe_avx2_i16_row_acc(6,  _kp, (A_i16), (lda_i16), _b, (c6)); \
            (c7) = fe_avx2_i16_row_acc(7,  _kp, (A_i16), (lda_i16), _b, (c7)); \
            (c8) = fe_avx2_i16_row_acc(8,  _kp, (A_i16), (lda_i16), _b, (c8)); \
            (c9) = fe_avx2_i16_row_acc(9,  _kp, (A_i16), (lda_i16), _b, (c9)); \
            (c10)= fe_avx2_i16_row_acc(10, _kp, (A_i16), (lda_i16), _b, (c10));\
            (c11)= fe_avx2_i16_row_acc(11, _kp, (A_i16), (lda_i16), _b, (c11));\
        }                                                                     \
    } while (0)

static inline void qgemm_kernel_12x8_avx2_i16(int K,
                                               const int16_t *A_i16, int lda_i16,
                                               const int8_t *Bp_kpair,
                                               int32_t *C32, int ldc32) {
    __m256i c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11;
    AVX2_I16_12_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                          c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
    _mm256_storeu_si256((__m256i *)(C32 + 0  * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1  * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2  * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3  * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4  * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5  * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6  * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7  * ldc32), c7);
    _mm256_storeu_si256((__m256i *)(C32 + 8  * ldc32), c8);
    _mm256_storeu_si256((__m256i *)(C32 + 9  * ldc32), c9);
    _mm256_storeu_si256((__m256i *)(C32 + 10 * ldc32), c10);
    _mm256_storeu_si256((__m256i *)(C32 + 11 * ldc32), c11);
}

/* vpmovsxbw+vpmaddwd SIMD narrow tail (1..11 rows). The MR=8 path had no tail
 * (every M was a multiple of 8); MR=12 introduces an M%12 tail (Winograd
 * NTiles=64 -> 4 rows). Routing those rows to a generic per-element scalar
 * kernel is catastrophic (scalar over Co*Ci per call, x24 Winograd calls/frame),
 * so the tail must stay on the vpmaddwd path against the i16-expanded A +
 * repacked B. */
static inline void qgemm_tail_avx2_i16_int32(int rows, int K,
                                             const int16_t *A_i16, int lda_i16,
                                             const int8_t *Bp_kpair,
                                             int32_t *C32, int ldc32) {
    int kp_count = K / 2;
    for (int r = 0; r < rows; ++r) {
        __m256i c = _mm256_setzero_si256();
        for (int kp = 0; kp < kp_count; ++kp) {
            __m256i b = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((const __m128i *)(Bp_kpair + (size_t)kp * 16)));
            c = fe_avx2_i16_row_acc(r, kp, A_i16, lda_i16, b, c);
        }
        _mm256_storeu_si256((__m256i *)(C32 + (size_t)r * ldc32), c);
    }
}

static inline void qgemm_kernel_8x8_avx2_i16(int K,
                                              const int16_t *A_i16, int lda_i16,
                                              const int8_t *Bp_kpair,
                                              int32_t *C32, int ldc32) {
    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
    AVX2_I16_8X8_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                            c0, c1, c2, c3, c4, c5, c6, c7);

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

/* Saturation-free signed×signed dot via vpsignb. ones16 is a hoisted
 * splat of int16(1); the caller is expected to keep it in a vector
 * register across the k-loop. */
static inline __m256i fe_avx2_row_acc(int R, int g,
                                      const int8_t *A_q, int lda,
                                      __m256i b_i8, __m256i ones16,
                                      __m256i acc) {
    __m256i a_i8 = fe_mem_broadcastd_epi32(A_q + R * lda + g * 4);
    __m256i ax   = _mm256_sign_epi8(a_i8, a_i8);   /* |a|, u8 ∈ [0,128] */
    __m256i sy   = _mm256_sign_epi8(b_i8, a_i8);   /* sign(a)*b */
    __m256i p16  = _mm256_maddubs_epi16(ax, sy);   /* saturation-free */
    __m256i p32  = _mm256_madd_epi16(p16, ones16); /* sum pairs → i32 */
    return _mm256_add_epi32(acc, p32);
}

static inline void qgemm_kernel_8x8_avx2(int K,
                                         const int8_t *A_q, int lda,
                                         const int8_t *Bp,
                                         int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx2_row_acc(0, g, A_q, lda, b, ones16, c0);
        c1 = fe_avx2_row_acc(1, g, A_q, lda, b, ones16, c1);
        c2 = fe_avx2_row_acc(2, g, A_q, lda, b, ones16, c2);
        c3 = fe_avx2_row_acc(3, g, A_q, lda, b, ones16, c3);
        c4 = fe_avx2_row_acc(4, g, A_q, lda, b, ones16, c4);
        c5 = fe_avx2_row_acc(5, g, A_q, lda, b, ones16, c5);
        c6 = fe_avx2_row_acc(6, g, A_q, lda, b, ones16, c6);
        c7 = fe_avx2_row_acc(7, g, A_q, lda, b, ones16, c7);
    }
    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

void qgemm_avx2_int32(int M, int N, int K,
                      const int8_t *A_q, const int8_t *Bp,
                      int32_t *C32, int ldc32) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;

    /* i16 path: pre-expand A to i16, re-pack B per N-block, then run
     * vpmovsxbw + vpmaddwd kernel. K must be even (HD_PAD_K=20 ensures this
     * for our model). Fall back to scalar/vpsignb path for tails. */
    int use_i16 = (K % 2 == 0) && ((size_t)M * (size_t)K <= FE_AVX2_I16_MAX_MK);
    if (use_i16) {
        fe_avx2_i16_expand_a(A_q, M, K, fe_avx2_a_i16_buf);
    }

    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        if (use_i16) {
            fe_avx2_i16_repack_b(B_block, k4_groups, fe_avx2_b_kpair_buf);
            int mr = 0;
            /* 24 = lcm(12-row tile, 8-row tile): capping the 12-row loop at the
             * largest multiple of 24 leaves an 8-aligned M remainder that tiles
             * exactly with the 8x8 kernel below — no scalar tail, no
             * fe_qgemm_tail_unsupported(). */
            for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 12-row tile; leave 8-aligned remainder */
                qgemm_kernel_12x8_avx2_i16(K,
                    fe_avx2_a_i16_buf + (size_t)mr * K, K,
                    fe_avx2_b_kpair_buf,
                    C32 + (size_t)mr * ldc32 + nr, ldc32);
            }
            for (; mr + 8 <= M; mr += 8) {
                qgemm_kernel_8x8_avx2_i16(K,
                    fe_avx2_a_i16_buf + (size_t)mr * K, K,
                    fe_avx2_b_kpair_buf,
                    C32 + (size_t)mr * ldc32 + nr, ldc32);
            }
            if (mr < M) {
                qgemm_tail_avx2_i16_int32(M - mr, K,
                    fe_avx2_a_i16_buf + (size_t)mr * K, K,
                    fe_avx2_b_kpair_buf,
                    C32 + (size_t)mr * ldc32 + nr, ldc32);
            }
        } else {
            int mr = 0;
            for (; mr + MR <= M; mr += MR) {
                qgemm_kernel_8x8_avx2(K, A_q + (size_t)mr * K, K,
                                      B_block,
                                      C32 + (size_t)mr * ldc32 + nr, ldc32);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/*
 * fp32-out fused dequant + bias (+ optional SiLU) at AVX2 width.
 * Op order: c32 -> fp32, then fmadd(bias, scale*c32), then optional SiLU.
 */
static inline void fe_avx2_fused_store_row_fp32(__m256i acc,
                                                __m256 vs,
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
qgemm_kernel_8x8_avx2_fp32(int K,
                           const int8_t *A_q, int lda,
                           const int8_t *Bp,
                           const float *combined_scale_n0,
                           const float *bias_n0,
                           float *C, int ldc,
                           int act_silu) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx2_row_acc(0, g, A_q, lda, b, ones16, c0);
        c1 = fe_avx2_row_acc(1, g, A_q, lda, b, ones16, c1);
        c2 = fe_avx2_row_acc(2, g, A_q, lda, b, ones16, c2);
        c3 = fe_avx2_row_acc(3, g, A_q, lda, b, ones16, c3);
        c4 = fe_avx2_row_acc(4, g, A_q, lda, b, ones16, c4);
        c5 = fe_avx2_row_acc(5, g, A_q, lda, b, ones16, c5);
        c6 = fe_avx2_row_acc(6, g, A_q, lda, b, ones16, c6);
        c7 = fe_avx2_row_acc(7, g, A_q, lda, b, ones16, c7);
    }
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avx2_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avx2_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avx2_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avx2_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avx2_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avx2_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avx2_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
}

static void
qgemm_kernel_8x8_avx2_fp32_i16(int K,
                                const int16_t *A_i16, int lda_i16,
                                const int8_t *Bp_kpair,
                                const float *combined_scale_n0,
                                const float *bias_n0,
                                float *C, int ldc,
                                int act_silu) {
    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
    AVX2_I16_8X8_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                            c0, c1, c2, c3, c4, c5, c6, c7);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32(c0, vs, bias_n0, act_silu, C + 0 * ldc);
    fe_avx2_fused_store_row_fp32(c1, vs, bias_n0, act_silu, C + 1 * ldc);
    fe_avx2_fused_store_row_fp32(c2, vs, bias_n0, act_silu, C + 2 * ldc);
    fe_avx2_fused_store_row_fp32(c3, vs, bias_n0, act_silu, C + 3 * ldc);
    fe_avx2_fused_store_row_fp32(c4, vs, bias_n0, act_silu, C + 4 * ldc);
    fe_avx2_fused_store_row_fp32(c5, vs, bias_n0, act_silu, C + 5 * ldc);
    fe_avx2_fused_store_row_fp32(c6, vs, bias_n0, act_silu, C + 6 * ldc);
    fe_avx2_fused_store_row_fp32(c7, vs, bias_n0, act_silu, C + 7 * ldc);
}

static void
qgemm_kernel_12x8_avx2_fp32_i16(int K,
                                const int16_t *A_i16, int lda_i16,
                                const int8_t *Bp_kpair,
                                const float *combined_scale_n0,
                                const float *bias_n0,
                                float *C, int ldc, int act_silu) {
    __m256i c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11;
    AVX2_I16_12_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                          c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32(c0,  vs, bias_n0, act_silu, C + 0  * ldc);
    fe_avx2_fused_store_row_fp32(c1,  vs, bias_n0, act_silu, C + 1  * ldc);
    fe_avx2_fused_store_row_fp32(c2,  vs, bias_n0, act_silu, C + 2  * ldc);
    fe_avx2_fused_store_row_fp32(c3,  vs, bias_n0, act_silu, C + 3  * ldc);
    fe_avx2_fused_store_row_fp32(c4,  vs, bias_n0, act_silu, C + 4  * ldc);
    fe_avx2_fused_store_row_fp32(c5,  vs, bias_n0, act_silu, C + 5  * ldc);
    fe_avx2_fused_store_row_fp32(c6,  vs, bias_n0, act_silu, C + 6  * ldc);
    fe_avx2_fused_store_row_fp32(c7,  vs, bias_n0, act_silu, C + 7  * ldc);
    fe_avx2_fused_store_row_fp32(c8,  vs, bias_n0, act_silu, C + 8  * ldc);
    fe_avx2_fused_store_row_fp32(c9,  vs, bias_n0, act_silu, C + 9  * ldc);
    fe_avx2_fused_store_row_fp32(c10, vs, bias_n0, act_silu, C + 10 * ldc);
    fe_avx2_fused_store_row_fp32(c11, vs, bias_n0, act_silu, C + 11 * ldc);
}


void qgemm_avx2_fp32_fused(int M, int N, int K,
                           const int8_t *A_q, const int8_t *Bp,
                           const float *combined_scale, const float *bias,
                           float *C, int ldc, int act_silu,
                           int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;

    int use_i16 = (K % 2 == 0) && ((size_t)M * (size_t)K <= FE_AVX2_I16_MAX_MK);
    if (use_i16) {
        fe_avx2_i16_expand_a(A_q, M, K, fe_avx2_a_i16_buf);
    }

    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        if (use_i16) {
            fe_avx2_i16_repack_b(B_block, k4_groups, fe_avx2_b_kpair_buf);
            const float *cs_n = combined_scale + nr;
            const float *bias_n = bias ? bias + nr : NULL;
            int mr = 0;
            for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 24 = lcm(12,8); see above */
                qgemm_kernel_12x8_avx2_fp32_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
            }
            for (; mr + 8 <= M; mr += 8) {
                qgemm_kernel_8x8_avx2_fp32_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, act_silu);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        } else {
            int mr = 0;
            for (; mr + MR <= M; mr += MR) {
                qgemm_kernel_8x8_avx2_fp32(
                    K, A_q + (size_t)mr * K, K, B_block,
                    combined_scale + nr,
                    bias ? bias + nr : NULL,
                    C + (size_t)mr * ldc + nr, ldc, act_silu);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32-out accumulating epilogue: C += bias + scale*c32 (FMA then add). */
static inline void fe_avx2_fused_store_row_fp32_acc(__m256i acc,
                                                    __m256 vs,
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
qgemm_kernel_8x8_avx2_fp32_acc(int K,
                               const int8_t *A_q, int lda,
                               const int8_t *Bp,
                               const float *combined_scale_n0,
                               const float *bias_n0,
                               float *C, int ldc) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx2_row_acc(0, g, A_q, lda, b, ones16, c0);
        c1 = fe_avx2_row_acc(1, g, A_q, lda, b, ones16, c1);
        c2 = fe_avx2_row_acc(2, g, A_q, lda, b, ones16, c2);
        c3 = fe_avx2_row_acc(3, g, A_q, lda, b, ones16, c3);
        c4 = fe_avx2_row_acc(4, g, A_q, lda, b, ones16, c4);
        c5 = fe_avx2_row_acc(5, g, A_q, lda, b, ones16, c5);
        c6 = fe_avx2_row_acc(6, g, A_q, lda, b, ones16, c6);
        c7 = fe_avx2_row_acc(7, g, A_q, lda, b, ones16, c7);
    }
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
}

static void
qgemm_kernel_8x8_avx2_fp32_acc_i16(int K,
                                    const int16_t *A_i16, int lda_i16,
                                    const int8_t *Bp_kpair,
                                    const float *combined_scale_n0,
                                    const float *bias_n0,
                                    float *C, int ldc) {
    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
    AVX2_I16_8X8_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                            c0, c1, c2, c3, c4, c5, c6, c7);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_acc(c0, vs, bias_n0, C + 0 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c1, vs, bias_n0, C + 1 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c2, vs, bias_n0, C + 2 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c3, vs, bias_n0, C + 3 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c4, vs, bias_n0, C + 4 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c5, vs, bias_n0, C + 5 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c6, vs, bias_n0, C + 6 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c7, vs, bias_n0, C + 7 * ldc);
}

static void
qgemm_kernel_12x8_avx2_fp32_acc_i16(int K,
                                    const int16_t *A_i16, int lda_i16,
                                    const int8_t *Bp_kpair,
                                    const float *combined_scale_n0,
                                    const float *bias_n0,
                                    float *C, int ldc) {
    __m256i c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11;
    AVX2_I16_12_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                          c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_acc(c0,  vs, bias_n0, C + 0  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c1,  vs, bias_n0, C + 1  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c2,  vs, bias_n0, C + 2  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c3,  vs, bias_n0, C + 3  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c4,  vs, bias_n0, C + 4  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c5,  vs, bias_n0, C + 5  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c6,  vs, bias_n0, C + 6  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c7,  vs, bias_n0, C + 7  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c8,  vs, bias_n0, C + 8  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c9,  vs, bias_n0, C + 9  * ldc);
    fe_avx2_fused_store_row_fp32_acc(c10, vs, bias_n0, C + 10 * ldc);
    fe_avx2_fused_store_row_fp32_acc(c11, vs, bias_n0, C + 11 * ldc);
}


__attribute__((hot))
void qgemm_avx2_fp32_fused_acc(int M, int N, int K,
                               const int8_t *A_q, const int8_t *Bp,
                               const float *combined_scale, const float *bias,
                               float *C, int ldc, int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;

    int use_i16 = (K % 2 == 0) && ((size_t)M * (size_t)K <= FE_AVX2_I16_MAX_MK);
    if (use_i16) {
        fe_avx2_i16_expand_a(A_q, M, K, fe_avx2_a_i16_buf);
    }

    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        if (use_i16) {
            fe_avx2_i16_repack_b(B_block, k4_groups, fe_avx2_b_kpair_buf);
            const float *cs_n = combined_scale + nr;
            const float *bias_n = bias ? bias + nr : NULL;
            int mr = 0;
            for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 24 = lcm(12,8); see above */
                qgemm_kernel_12x8_avx2_fp32_acc_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
            }
            for (; mr + 8 <= M; mr += 8) {
                qgemm_kernel_8x8_avx2_fp32_acc_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        } else {
            int mr = 0;
            for (; mr + MR <= M; mr += MR) {
                qgemm_kernel_8x8_avx2_fp32_acc(
                    K, A_q + (size_t)mr * K, K, B_block,
                    combined_scale + nr,
                    bias ? bias + nr : NULL,
                    C + (size_t)mr * ldc + nr, ldc);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
}

/* fp32-out with max-abs tracking (no SiLU). */
static inline void fe_avx2_fused_store_row_fp32_track(__m256i acc,
                                                      __m256 vs,
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
qgemm_kernel_8x8_avx2_fp32_track(int K,
                                 const int8_t *A_q, int lda,
                                 const int8_t *Bp,
                                 const float *combined_scale_n0,
                                 const float *bias_n0,
                                 float *C, int ldc,
                                 __m256 *vmax) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(Bp + (size_t)g * 32));
        c0 = fe_avx2_row_acc(0, g, A_q, lda, b, ones16, c0);
        c1 = fe_avx2_row_acc(1, g, A_q, lda, b, ones16, c1);
        c2 = fe_avx2_row_acc(2, g, A_q, lda, b, ones16, c2);
        c3 = fe_avx2_row_acc(3, g, A_q, lda, b, ones16, c3);
        c4 = fe_avx2_row_acc(4, g, A_q, lda, b, ones16, c4);
        c5 = fe_avx2_row_acc(5, g, A_q, lda, b, ones16, c5);
        c6 = fe_avx2_row_acc(6, g, A_q, lda, b, ones16, c6);
        c7 = fe_avx2_row_acc(7, g, A_q, lda, b, ones16, c7);
    }
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
}

static void
qgemm_kernel_12x8_avx2_fp32_track_i16(int K,
                                      const int16_t *A_i16, int lda_i16,
                                      const int8_t *Bp_kpair,
                                      const float *combined_scale_n0,
                                      const float *bias_n0,
                                      float *C, int ldc, __m256 *vmax) {
    __m256i c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11;
    AVX2_I16_12_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                          c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_track(c0,  vs, bias_n0, C + 0  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c1,  vs, bias_n0, C + 1  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c2,  vs, bias_n0, C + 2  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c3,  vs, bias_n0, C + 3  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c4,  vs, bias_n0, C + 4  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c5,  vs, bias_n0, C + 5  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c6,  vs, bias_n0, C + 6  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c7,  vs, bias_n0, C + 7  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c8,  vs, bias_n0, C + 8  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c9,  vs, bias_n0, C + 9  * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c10, vs, bias_n0, C + 10 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c11, vs, bias_n0, C + 11 * ldc, vmax);
}

static void
qgemm_kernel_8x8_avx2_fp32_track_i16(int K,
                                      const int16_t *A_i16, int lda_i16,
                                      const int8_t *Bp_kpair,
                                      const float *combined_scale_n0,
                                      const float *bias_n0,
                                      float *C, int ldc,
                                      __m256 *vmax) {
    __m256i c0, c1, c2, c3, c4, c5, c6, c7;
    AVX2_I16_8X8_TILE_BODY(K, A_i16, lda_i16, Bp_kpair,
                            c0, c1, c2, c3, c4, c5, c6, c7);

    __m256 vs = _mm256_loadu_ps(combined_scale_n0);
    fe_avx2_fused_store_row_fp32_track(c0, vs, bias_n0, C + 0 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c1, vs, bias_n0, C + 1 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c2, vs, bias_n0, C + 2 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c3, vs, bias_n0, C + 3 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c4, vs, bias_n0, C + 4 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c5, vs, bias_n0, C + 5 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c6, vs, bias_n0, C + 6 * ldc, vmax);
    fe_avx2_fused_store_row_fp32_track(c7, vs, bias_n0, C + 7 * ldc, vmax);
}

__attribute__((hot))
void qgemm_avx2_fp32_fused_track_maxabs(int M, int N, int K,
                                        const int8_t *A_q, const int8_t *Bp,
                                        const float *combined_scale,
                                        const float *bias,
                                        float *C, int ldc,
                                        int32_t *c32_tail,
                                        float *max_abs_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    __m256 vmax = _mm256_setzero_ps();

    int use_i16 = (K % 2 == 0) && ((size_t)M * (size_t)K <= FE_AVX2_I16_MAX_MK);
    if (use_i16) {
        fe_avx2_i16_expand_a(A_q, M, K, fe_avx2_a_i16_buf);
    }

    int nr = 0, bn = 0;
    for (; nr + NR <= N; nr += NR, ++bn) {
        const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
        if (use_i16) {
            fe_avx2_i16_repack_b(B_block, k4_groups, fe_avx2_b_kpair_buf);
            const float *cs_n = combined_scale + nr;
            const float *bias_n = bias ? bias + nr : NULL;
            int mr = 0;
            for (; mr + 12 <= (M / 24) * 24; mr += 12) {  /* 24 = lcm(12,8); see above */
                qgemm_kernel_12x8_avx2_fp32_track_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, &vmax);
            }
            for (; mr + 8 <= M; mr += 8) {
                qgemm_kernel_8x8_avx2_fp32_track_i16(
                    K, fe_avx2_a_i16_buf + (size_t)mr * K, K, fe_avx2_b_kpair_buf,
                    cs_n, bias_n, C + (size_t)mr * ldc + nr, ldc, &vmax);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        } else {
            int mr = 0;
            for (; mr + MR <= M; mr += MR) {
                qgemm_kernel_8x8_avx2_fp32_track(
                    K, A_q + (size_t)mr * K, K, B_block,
                    combined_scale + nr,
                    bias ? bias + nr : NULL,
                    C + (size_t)mr * ldc + nr, ldc, &vmax);
            }
            if (mr < M) fe_qgemm_tail_unsupported();
        }
    }
    if (nr < N) fe_qgemm_tail_unsupported();
    /* Horizontal max-reduce ymm to scalar. */
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 mx = _mm_max_ps(lo, hi);
    mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
    mx = _mm_max_ss(mx, _mm_shuffle_ps(mx, mx, 0x55));
    _mm_store_ss(max_abs_out, mx);
}

/* K=20 specialisation: k4_groups=5 fully unrolled. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x8_avx2_k20(const int8_t *A_q, int lda,
                                             const int8_t *Bp,
                                             int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);

    #define AVX2_K20_GROUP(G) do {                                          \
        __m256i b = _mm256_loadu_si256(                                     \
            (const __m256i *)(Bp + (size_t)(G) * 32));                      \
        c0 = fe_avx2_row_acc(0, (G), A_q, lda, b, ones16, c0);              \
        c1 = fe_avx2_row_acc(1, (G), A_q, lda, b, ones16, c1);              \
        c2 = fe_avx2_row_acc(2, (G), A_q, lda, b, ones16, c2);              \
        c3 = fe_avx2_row_acc(3, (G), A_q, lda, b, ones16, c3);              \
        c4 = fe_avx2_row_acc(4, (G), A_q, lda, b, ones16, c4);              \
        c5 = fe_avx2_row_acc(5, (G), A_q, lda, b, ones16, c5);              \
        c6 = fe_avx2_row_acc(6, (G), A_q, lda, b, ones16, c6);              \
        c7 = fe_avx2_row_acc(7, (G), A_q, lda, b, ones16, c7);              \
    } while (0)
    AVX2_K20_GROUP(0);
    AVX2_K20_GROUP(1);
    AVX2_K20_GROUP(2);
    AVX2_K20_GROUP(3);
    AVX2_K20_GROUP(4);
    #undef AVX2_K20_GROUP

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

/* k20 specialisation: 10 K-pairs fully unrolled, vpmovsxbw+vpmaddwd. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x8_avx2_k20_i16(const int16_t *A_i16, int lda_i16,
                                                  const int8_t *Bp_kpair,
                                                  int32_t *C32, int ldc32) {
    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(), c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(), c7 = _mm256_setzero_si256();

    #define AVX2_K20_I16_PAIR(KP) do {                                        \
        __m256i b = _mm256_cvtepi8_epi16(                                     \
            _mm_loadu_si128((const __m128i *)(Bp_kpair + (size_t)(KP) * 16))); \
        c0 = fe_avx2_i16_row_acc(0, (KP), A_i16, lda_i16, b, c0);             \
        c1 = fe_avx2_i16_row_acc(1, (KP), A_i16, lda_i16, b, c1);             \
        c2 = fe_avx2_i16_row_acc(2, (KP), A_i16, lda_i16, b, c2);             \
        c3 = fe_avx2_i16_row_acc(3, (KP), A_i16, lda_i16, b, c3);             \
        c4 = fe_avx2_i16_row_acc(4, (KP), A_i16, lda_i16, b, c4);             \
        c5 = fe_avx2_i16_row_acc(5, (KP), A_i16, lda_i16, b, c5);             \
        c6 = fe_avx2_i16_row_acc(6, (KP), A_i16, lda_i16, b, c6);             \
        c7 = fe_avx2_i16_row_acc(7, (KP), A_i16, lda_i16, b, c7);             \
    } while (0)
    AVX2_K20_I16_PAIR(0); AVX2_K20_I16_PAIR(1); AVX2_K20_I16_PAIR(2);
    AVX2_K20_I16_PAIR(3); AVX2_K20_I16_PAIR(4); AVX2_K20_I16_PAIR(5);
    AVX2_K20_I16_PAIR(6); AVX2_K20_I16_PAIR(7); AVX2_K20_I16_PAIR(8);
    AVX2_K20_I16_PAIR(9);
    #undef AVX2_K20_I16_PAIR

    _mm256_storeu_si256((__m256i *)(C32 + 0 * ldc32), c0);
    _mm256_storeu_si256((__m256i *)(C32 + 1 * ldc32), c1);
    _mm256_storeu_si256((__m256i *)(C32 + 2 * ldc32), c2);
    _mm256_storeu_si256((__m256i *)(C32 + 3 * ldc32), c3);
    _mm256_storeu_si256((__m256i *)(C32 + 4 * ldc32), c4);
    _mm256_storeu_si256((__m256i *)(C32 + 5 * ldc32), c5);
    _mm256_storeu_si256((__m256i *)(C32 + 6 * ldc32), c6);
    _mm256_storeu_si256((__m256i *)(C32 + 7 * ldc32), c7);
}

void qgemm_avx2_int32_k20(int M, int N,
                          const int8_t *A_q, const int8_t *Bp,
                          int32_t *C32, int ldc32) {
    enum { K = 20, MR = FE_QGEMM_MR, NR = FE_QGEMM_NR };
    enum { k4_groups = (K + 3) / 4 };

    int use_i16 = ((size_t)M * (size_t)K <= FE_AVX2_I16_MAX_MK);
    if (use_i16) {
        fe_avx2_i16_expand_a(A_q, M, K, fe_avx2_a_i16_buf);
    }

    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            const int8_t *B_block = Bp + (size_t)bn * k4_groups * NR * 4;
            if (use_i16) {
                fe_avx2_i16_repack_b(B_block, k4_groups, fe_avx2_b_kpair_buf);
                qgemm_kernel_8x8_avx2_k20_i16(
                    fe_avx2_a_i16_buf + (size_t)mr * K, K,
                    fe_avx2_b_kpair_buf,
                    C32 + (size_t)mr * ldc32 + nr, ldc32);
            } else {
                qgemm_kernel_8x8_avx2_k20(A_q + (size_t)mr * K, K,
                                           B_block,
                                           C32 + (size_t)mr * ldc32 + nr, ldc32);
            }
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/*
 * GRU full-fused W_ih + W_hh + gate helpers. Uses fe_qg_sigmoid8 /
 * fe_qg_tanh8 at ymm width.
 */
static inline void fe_avx2_rzgate_row(__m256i acc,
                                       __m256 vs, __m256 vb,
                                       const float *ih_row,
                                       const float *bsum_n,
                                       float *band, int n_off, int ld_band,
                                       int r_idx) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    /* hh = bias_eff + scale * c32 */
    __m256 hh = _mm256_fmadd_ps(vc, vs, vb);
    __m256 pre = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(ih_row + n_off), hh),
                               _mm256_loadu_ps(bsum_n));
    __m256 g = fe_qg_sigmoid8(pre);
    _mm256_storeu_ps(band + r_idx * ld_band + n_off, g);
}


#define AVX2_8X8_TILE_BODY(K, A_q, lda, Bp,                                  \
                           c0, c1, c2, c3, c4, c5, c6, c7)                   \
    do {                                                                      \
        c0 = c1 = c2 = c3 = _mm256_setzero_si256();                          \
        c4 = c5 = c6 = c7 = _mm256_setzero_si256();                          \
        const __m256i _ones16 = _mm256_set1_epi16(1);                        \
        int _k4g = (K) / 4;                                                  \
        for (int _g = 0; _g < _k4g; ++_g) {                                  \
            __m256i _b = _mm256_loadu_si256(                                  \
                (const __m256i *)((Bp) + (size_t)_g * 32));                   \
            c0 = fe_avx2_row_acc(0, _g, (A_q), (lda), _b, _ones16, c0);      \
            c1 = fe_avx2_row_acc(1, _g, (A_q), (lda), _b, _ones16, c1);      \
            c2 = fe_avx2_row_acc(2, _g, (A_q), (lda), _b, _ones16, c2);      \
            c3 = fe_avx2_row_acc(3, _g, (A_q), (lda), _b, _ones16, c3);      \
            c4 = fe_avx2_row_acc(4, _g, (A_q), (lda), _b, _ones16, c4);      \
            c5 = fe_avx2_row_acc(5, _g, (A_q), (lda), _b, _ones16, c5);      \
            c6 = fe_avx2_row_acc(6, _g, (A_q), (lda), _b, _ones16, c6);      \
            c7 = fe_avx2_row_acc(7, _g, (A_q), (lda), _b, _ones16, c7);      \
        }                                                                     \
    } while (0)


/*
 * Full-fused GRU: fuse W_ih @ x, W_hh @ h, and the gate update in one
 * row-block pass.
 */
static inline void fe_avx2_dq_row_to_ih(__m256i acc,
                                         __m256 vs, __m256 vb,
                                         float *ih_tile_row) {
    __m256 vc = _mm256_cvtepi32_ps(acc);
    __m256 ih = _mm256_fmadd_ps(vc, vs, vb);
    _mm256_storeu_ps(ih_tile_row, ih);
}

/* GRU scratch: pre-expanded X and H in i16, pre-packed Wq_ih and
 * Wq_hh in K-pair-interleaved form. Sized for D <= FE_QGEMM_MAX_GRU_D
 * (128). BSS cost ~160 KiB, only touched on AVX2 tier. */
static FE_ALIGN64 int16_t fe_avx2_gru_x_i16[FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D];
static FE_ALIGN64 int16_t fe_avx2_gru_h_i16[FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D];
static FE_ALIGN64 int8_t  fe_avx2_gru_wih_kpair[3 * FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D];
static FE_ALIGN64 int8_t  fe_avx2_gru_whh_kpair[3 * FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D];

/* peak-stabilization: pre-fault all AVX2 BSS scratch pages at
 * fe_qgemm_init. The i16/K-pair scratch buffers live in BSS, which Windows/Linux
 * lazily back with COW-zero until first write — each fresh page causes
 * a minor page fault on first touch. Frame-200..1999 percentiles can
 * still catch occasional faults if a page is first reached only by a
 * later-layer/tail path. Writing one byte per page up-front commits all
 * pages once, taking the per-page fault cost out of steady-state
 * percentiles. */
void qgemm_avx2_prefault_buffers(void) {
    const size_t PAGE = 4096;
    volatile unsigned char *p;

    p = (volatile unsigned char *)fe_avx2_a_i16_buf;
    for (size_t i = 0; i < sizeof(fe_avx2_a_i16_buf); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_avx2_b_kpair_buf;
    for (size_t i = 0; i < sizeof(fe_avx2_b_kpair_buf); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_avx2_gru_x_i16;
    for (size_t i = 0; i < sizeof(fe_avx2_gru_x_i16); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_avx2_gru_h_i16;
    for (size_t i = 0; i < sizeof(fe_avx2_gru_h_i16); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_avx2_gru_wih_kpair;
    for (size_t i = 0; i < sizeof(fe_avx2_gru_wih_kpair); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_avx2_gru_whh_kpair;
    for (size_t i = 0; i < sizeof(fe_avx2_gru_whh_kpair); i += PAGE) p[i] = 0;
}


/* AVX2 fp16 inout variant (gru_h stored fp16; ngate epilogue writes state
 * back as fp16). Dual-store ngate_row eliminates both engine unpack and
 * pack passes. */
__attribute__((target("avx2,f16c,fma")))
static inline void fe_avx2_ngate_row_fp16inout(__m256i acc,
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

__attribute__((hot))
__attribute__((target("avx2,f16c,fma")))
void qgemm_avx2_gru_full_fused_fp16inout(int M, int D,
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
    const int MR = FE_QGEMM_GRU_MR;  /* 12-row tile */
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

    /* The engine's fp16-inout GRU previously ran the slower vpsignb path
     * (5 inner ops/row/K-quartet). Promote it to the vpmovsxbw + vpmaddwd
     * path (2 inner ops) used by every other AVX2 kernel: pre-expand X/H to
     * i16 + pre-pack all W to K-pair once per call (hoisted over all
     * M-blocks). Bit-id: vpmaddwd yields the identical signed int32 dot as
     * the vpsignb+vpmaddubsw chain. */
    for (int m = 0; m < M; ++m) {
        const int8_t *src_x = Xq + (size_t)m * ld_x;
        const int8_t *src_h = Hq + (size_t)m * ld_h;
        int16_t *dst_x = fe_avx2_gru_x_i16 + (size_t)m * K;
        int16_t *dst_h = fe_avx2_gru_h_i16 + (size_t)m * K;
        int k = 0;
        for (; k + 15 < K; k += 16) {
            _mm256_storeu_si256((__m256i *)(dst_x + k),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(src_x + k))));
            _mm256_storeu_si256((__m256i *)(dst_h + k),
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(src_h + k))));
        }
        for (; k < K; ++k) { dst_x[k] = (int16_t)src_x[k]; dst_h[k] = (int16_t)src_h[k]; }
    }
    {
        int total_blocks = 3 * N_blocks_per_gate;
        for (int bidx = 0; bidx < total_blocks; ++bidx) {
            fe_avx2_i16_repack_b(Wq_ih + (size_t)bidx * weight_n_block_stride,
                k4_groups, fe_avx2_gru_wih_kpair + (size_t)bidx * weight_n_block_stride);
            fe_avx2_i16_repack_b(Wq_hh + (size_t)bidx * weight_n_block_stride,
                k4_groups, fe_avx2_gru_whh_kpair + (size_t)bidx * weight_n_block_stride);
        }
    }

    __m256i c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;

    for (int mr = 0; mr + MR <= M; mr += MR) {
        const int16_t *X_i16_block = fe_avx2_gru_x_i16 + (size_t)mr * K;
        const int16_t *H_i16_block = fe_avx2_gru_h_i16 + (size_t)mr * K;
        uint16_t     *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float        *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;

        for (int slice = 0; slice < 3; ++slice) {
            const int gate_off = slice * D;
            for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
                int n_off = nb * NR;
                const int8_t *Bp_kp = fe_avx2_gru_wih_kpair +
                    (size_t)(slice * N_blocks_per_gate + nb) * weight_n_block_stride;
                AVX2_I16_12_TILE_BODY(K, X_i16_block, K, Bp_kp,
                                      c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
                __m256 vs = _mm256_loadu_ps(combined_ih + gate_off + n_off);
                __m256 vb = _mm256_loadu_ps(bias_eff_ih + gate_off + n_off);
                float *row = ih_tile + gate_off + n_off;
                fe_avx2_dq_row_to_ih(c0, vs, vb, row + 0 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c1, vs, vb, row + 1 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c2, vs, vb, row + 2 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c3, vs, vb, row + 3 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c4, vs, vb, row + 4 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c5, vs, vb, row + 5 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c6, vs, vb, row + 6 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c7, vs, vb, row + 7 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c8, vs, vb, row + 8 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c9, vs, vb, row + 9 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c10, vs, vb, row + 10 * ld_ih_tile);
                fe_avx2_dq_row_to_ih(c11, vs, vb, row + 11 * ld_ih_tile);
            }
        }

        const float *ih_block = ih_tile;

        /* r gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            int n_off = nb * NR;
            const int8_t *Bp_kp = fe_avx2_gru_whh_kpair + (size_t)nb * weight_n_block_stride;
            AVX2_I16_12_TILE_BODY(K, H_i16_block, K, Bp_kp, c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + n_off);
            fe_avx2_rzgate_row(c0, vs, vb, ih_block + 0 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 0);
            fe_avx2_rzgate_row(c1, vs, vb, ih_block + 1 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 1);
            fe_avx2_rzgate_row(c2, vs, vb, ih_block + 2 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 2);
            fe_avx2_rzgate_row(c3, vs, vb, ih_block + 3 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 3);
            fe_avx2_rzgate_row(c4, vs, vb, ih_block + 4 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 4);
            fe_avx2_rzgate_row(c5, vs, vb, ih_block + 5 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 5);
            fe_avx2_rzgate_row(c6, vs, vb, ih_block + 6 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 6);
            fe_avx2_rzgate_row(c7, vs, vb, ih_block + 7 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 7);
            fe_avx2_rzgate_row(c8, vs, vb, ih_block + 8 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 8);
            fe_avx2_rzgate_row(c9, vs, vb, ih_block + 9 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 9);
            fe_avx2_rzgate_row(c10, vs, vb, ih_block + 10 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 10);
            fe_avx2_rzgate_row(c11, vs, vb, ih_block + 11 * ld_ih_tile, br_sum + n_off, r_band, n_off, ld_band, 11);
        }

        /* z gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            int n_off = nb * NR;
            const int8_t *Bp_kp = fe_avx2_gru_whh_kpair +
                (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            AVX2_I16_12_TILE_BODY(K, H_i16_block, K, Bp_kp, c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + D + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + D + n_off);
            const float *ih_z = ih_block + D;
            fe_avx2_rzgate_row(c0, vs, vb, ih_z + 0 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 0);
            fe_avx2_rzgate_row(c1, vs, vb, ih_z + 1 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 1);
            fe_avx2_rzgate_row(c2, vs, vb, ih_z + 2 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 2);
            fe_avx2_rzgate_row(c3, vs, vb, ih_z + 3 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 3);
            fe_avx2_rzgate_row(c4, vs, vb, ih_z + 4 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 4);
            fe_avx2_rzgate_row(c5, vs, vb, ih_z + 5 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 5);
            fe_avx2_rzgate_row(c6, vs, vb, ih_z + 6 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 6);
            fe_avx2_rzgate_row(c7, vs, vb, ih_z + 7 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 7);
            fe_avx2_rzgate_row(c8, vs, vb, ih_z + 8 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 8);
            fe_avx2_rzgate_row(c9, vs, vb, ih_z + 9 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 9);
            fe_avx2_rzgate_row(c10, vs, vb, ih_z + 10 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 10);
            fe_avx2_rzgate_row(c11, vs, vb, ih_z + 11 * ld_ih_tile, bz_sum + n_off, z_band, n_off, ld_band, 11);
        }

        /* n gate + h_new — fp16 read, fp32 write. */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            int n_off = nb * NR;
            const int8_t *Bp_kp = fe_avx2_gru_whh_kpair +
                (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            AVX2_I16_12_TILE_BODY(K, H_i16_block, K, Bp_kp, c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11);
            __m256 vs = _mm256_loadu_ps(combined_hh + 2 * D + n_off);
            __m256 vb = _mm256_loadu_ps(bias_eff_hh + 2 * D + n_off);
            const float *ih_n = ih_block + 2 * D;
            fe_avx2_ngate_row_fp16inout(c0, vs, vb, ih_n + 0 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 0 * ld_h_inout, h_block_scratch + 0 * ld_h_out, 0);
            fe_avx2_ngate_row_fp16inout(c1, vs, vb, ih_n + 1 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 1 * ld_h_inout, h_block_scratch + 1 * ld_h_out, 1);
            fe_avx2_ngate_row_fp16inout(c2, vs, vb, ih_n + 2 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 2 * ld_h_inout, h_block_scratch + 2 * ld_h_out, 2);
            fe_avx2_ngate_row_fp16inout(c3, vs, vb, ih_n + 3 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 3 * ld_h_inout, h_block_scratch + 3 * ld_h_out, 3);
            fe_avx2_ngate_row_fp16inout(c4, vs, vb, ih_n + 4 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 4 * ld_h_inout, h_block_scratch + 4 * ld_h_out, 4);
            fe_avx2_ngate_row_fp16inout(c5, vs, vb, ih_n + 5 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 5 * ld_h_inout, h_block_scratch + 5 * ld_h_out, 5);
            fe_avx2_ngate_row_fp16inout(c6, vs, vb, ih_n + 6 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 6 * ld_h_inout, h_block_scratch + 6 * ld_h_out, 6);
            fe_avx2_ngate_row_fp16inout(c7, vs, vb, ih_n + 7 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 7 * ld_h_inout, h_block_scratch + 7 * ld_h_out, 7);
            fe_avx2_ngate_row_fp16inout(c8, vs, vb, ih_n + 8 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 8 * ld_h_inout, h_block_scratch + 8 * ld_h_out, 8);
            fe_avx2_ngate_row_fp16inout(c9, vs, vb, ih_n + 9 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 9 * ld_h_inout, h_block_scratch + 9 * ld_h_out, 9);
            fe_avx2_ngate_row_fp16inout(c10, vs, vb, ih_n + 10 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 10 * ld_h_inout, h_block_scratch + 10 * ld_h_out, 10);
            fe_avx2_ngate_row_fp16inout(c11, vs, vb, ih_n + 11 * ld_ih_tile, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 11 * ld_h_inout, h_block_scratch + 11 * ld_h_out, 11);
        }
    }
}

#undef AVX2_8X8_TILE_BODY


#endif /* x86 */

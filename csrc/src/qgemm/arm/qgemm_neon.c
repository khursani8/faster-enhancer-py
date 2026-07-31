/*
 * arm/qgemm_neon.c - ARMv8.0 NEON baseline int8 GEMM (no DOTPROD/I8MM).
 *
 * Uses vmull_s8 + vpaddlq_s16 + vpaddq_s32 for the 8x8 microkernel with K
 * multiple of 4. Fused epilogues match the 2-pass dequant FMA order for
 * bit-identical results.
 */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../../internal/fe_simd.h"   /* FE_SIGMOIDF4 for SiLU */
#include "../../internal/fe_fp16.h"   /* fp16 inout: load/store helpers */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* vpadalq-fused row accumulate (NEON baseline acceleration).
 *
 * Borrows the ggml q3_K / tq1_0 pattern: vmull_s8 → vpadalq_s16 saves
 * the separate vpaddlq_s16 + vaddq_s32 pair into a single fused
 * pair-add-accumulate. Layout of a row's two i32x4 partial accumulators
 * (p_lo, p_hi), with b_4cols a 16-byte block holding 4 cols × 4 K-lanes:
 *
 *   plo = vmull_s8(a, vget_low_s8(b))   → 8 pair-products from cols 0,1
 *   phi = vmull_s8(a, vget_high_s8(b))  → 8 pair-products from cols 2,3
 *   p_lo = vpadalq_s16(p_lo, plo)       → [c0_pA, c0_pB, c1_pA, c1_pB]
 *   p_hi = vpadalq_s16(p_hi, phi)       → [c2_pA, c2_pB, c3_pA, c3_pB]
 *
 * After K-loop, vpaddq_s32(p_lo, p_hi) yields [c0, c1, c2, c3] in one
 * instruction. Op count per row per K-group: 4 (vs 6 of widening path).
 */
static inline void fe_neon_row_acc_vpadalq(int R, int g,
                                            const int8_t *A_q, int lda,
                                            int8x16_t b_4cols,
                                            int32x4_t *p_lo, int32x4_t *p_hi) {
    int8x8_t a4_bc = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(A_q + R * lda + g * 4)));
    int16x8_t plo = vmull_s8(a4_bc, vget_low_s8(b_4cols));
    int16x8_t phi = vmull_s8(a4_bc, vget_high_s8(b_4cols));
    *p_lo = vpadalq_s16(*p_lo, plo);
    *p_hi = vpadalq_s16(*p_hi, phi);
}

/* 2-K-step vmlal-chain row accumulator (NEON op-count reduction step 2).
 *
 * Borrows the ggml q3_K i16-level vmlal_s8 chain idea. With both weight
 * and activation clamped to [-127, 127] (universal weight/act clamp), each i16
 * lane holds at most 2 × 127×127 = 32,258 after two vmlal_s8 chained
 * onto a vmull_s8 start — safe under i16 max ±32,767 by margin 509.
 * A third chained vmlal would overflow.
 *
 * Per 2 K-groups per row (4 cols): 2 vmull + 2 vmlal + 2 vpadalq = 6 ops
 * vs 8 ops of single-step vpadalq path. 25% inner-loop op reduction on
 * top of the existing −33% from the vmull→vpadalq fusion. Math is
 * associative at i32 level → byte-identical to the single-step path.
 */
static inline void fe_neon_row_acc_vmlal2(int R, int g,
                                           const int8_t *A_q, int lda,
                                           int8x16_t b0, int8x16_t b1,
                                           int32x4_t *p_lo, int32x4_t *p_hi) {
    int8x8_t a0_bc = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(A_q + R * lda + g * 4)));
    int8x8_t a1_bc = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(A_q + R * lda + (g + 1) * 4)));
    /* Cols 0-1: vmull (K-group g) → vmlal (K-group g+1) → i16 partial */
    int16x8_t plo = vmull_s8(a0_bc, vget_low_s8(b0));
    plo = vmlal_s8(plo, a1_bc, vget_low_s8(b1));
    /* Cols 2-3: same shape */
    int16x8_t phi = vmull_s8(a0_bc, vget_high_s8(b0));
    phi = vmlal_s8(phi, a1_bc, vget_high_s8(b1));
    /* Single promote-and-accumulate (pair-sum i16 → i32) */
    *p_lo = vpadalq_s16(*p_lo, plo);
    *p_hi = vpadalq_s16(*p_hi, phi);
}

/* 4x8 (not 8x4-twice): 8-column row accumulators. A row's 4-byte K-quad is loaded ONCE and
 * reused across all 8 output columns (cols 0-3 from b*lo, cols 4-7 from
 * b*hi), instead of the 8x4-twice path that re-loads A for the second
 * 4-col sub-block. Halves A-side LD1R traffic — the dominant cost on
 * load-narrow NEON cores (Cortex-A55). Same vmull/vmlal/vpadalq ops in the
 * same order as two fe_neon_row_acc_vmlal2 calls → bit-identical C32. */
static inline void fe_neon_row8_acc_vmlal2(const int8_t *Arow, int g,
                                            int8x16_t b0lo, int8x16_t b0hi,
                                            int8x16_t b1lo, int8x16_t b1hi,
                                            int32x4_t *plo, int32x4_t *phi,
                                            int32x4_t *Plo, int32x4_t *Phi) {
    int8x8_t a0 = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(Arow + (size_t)g * 4)));
    int8x8_t a1 = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(Arow + (size_t)(g + 1) * 4)));
    int16x8_t lo = vmull_s8(a0, vget_low_s8(b0lo));
    lo = vmlal_s8(lo, a1, vget_low_s8(b1lo));
    int16x8_t hi = vmull_s8(a0, vget_high_s8(b0lo));
    hi = vmlal_s8(hi, a1, vget_high_s8(b1lo));
    int16x8_t Lo = vmull_s8(a0, vget_low_s8(b0hi));
    Lo = vmlal_s8(Lo, a1, vget_low_s8(b1hi));
    int16x8_t Hi = vmull_s8(a0, vget_high_s8(b0hi));
    Hi = vmlal_s8(Hi, a1, vget_high_s8(b1hi));
    *plo = vpadalq_s16(*plo, lo);  *phi = vpadalq_s16(*phi, hi);
    *Plo = vpadalq_s16(*Plo, Lo);  *Phi = vpadalq_s16(*Phi, Hi);
}

/* Single-k4-group tail (8 cols, A loaded once). */
static inline void fe_neon_row8_acc_vpadalq(const int8_t *Arow, int g,
                                            int8x16_t blo, int8x16_t bhi,
                                            int32x4_t *plo, int32x4_t *phi,
                                            int32x4_t *Plo, int32x4_t *Phi) {
    int8x8_t a = vreinterpret_s8_s32(
        vld1_dup_s32((const int32_t *)(Arow + (size_t)g * 4)));
    *plo = vpadalq_s16(*plo, vmull_s8(a, vget_low_s8(blo)));
    *phi = vpadalq_s16(*phi, vmull_s8(a, vget_high_s8(blo)));
    *Plo = vpadalq_s16(*Plo, vmull_s8(a, vget_low_s8(bhi)));
    *Phi = vpadalq_s16(*Phi, vmull_s8(a, vget_high_s8(bhi)));
}

/* 4-row × 8-col macro: run the K loop accumulating into 16 partials
 * (4 rows × {plo,phi,Plo,Phi}). Bp k4-group stride is NR=8 group = 32 B;
 * cols 0-3 are bytes 0-15 (lo), cols 4-7 are bytes 16-31 (hi). */
#define FE_NEON_4x8_KLOOP(K, A_q, lda, Bp,                                    \
                          q00,q01,q02,q03, q10,q11,q12,q13,                   \
                          q20,q21,q22,q23, q30,q31,q32,q33)                   \
    do {                                                                      \
        int _kg = (K) / 4, _g = 0;                                            \
        for (; _g + 1 < _kg; _g += 2) {                                       \
            int8x16_t _b0lo = vld1q_s8((Bp) + (size_t)_g * 32);               \
            int8x16_t _b0hi = vld1q_s8((Bp) + (size_t)_g * 32 + 16);          \
            int8x16_t _b1lo = vld1q_s8((Bp) + (size_t)(_g + 1) * 32);         \
            int8x16_t _b1hi = vld1q_s8((Bp) + (size_t)(_g + 1) * 32 + 16);    \
            fe_neon_row8_acc_vmlal2((A_q) + 0 * (lda), _g, _b0lo,_b0hi,_b1lo,_b1hi, &q00,&q01,&q02,&q03); \
            fe_neon_row8_acc_vmlal2((A_q) + 1 * (lda), _g, _b0lo,_b0hi,_b1lo,_b1hi, &q10,&q11,&q12,&q13); \
            fe_neon_row8_acc_vmlal2((A_q) + 2 * (lda), _g, _b0lo,_b0hi,_b1lo,_b1hi, &q20,&q21,&q22,&q23); \
            fe_neon_row8_acc_vmlal2((A_q) + 3 * (lda), _g, _b0lo,_b0hi,_b1lo,_b1hi, &q30,&q31,&q32,&q33); \
        }                                                                     \
        if (_g < _kg) {                                                       \
            int8x16_t _blo = vld1q_s8((Bp) + (size_t)_g * 32);                \
            int8x16_t _bhi = vld1q_s8((Bp) + (size_t)_g * 32 + 16);           \
            fe_neon_row8_acc_vpadalq((A_q) + 0 * (lda), _g, _blo,_bhi, &q00,&q01,&q02,&q03); \
            fe_neon_row8_acc_vpadalq((A_q) + 1 * (lda), _g, _blo,_bhi, &q10,&q11,&q12,&q13); \
            fe_neon_row8_acc_vpadalq((A_q) + 2 * (lda), _g, _blo,_bhi, &q20,&q21,&q22,&q23); \
            fe_neon_row8_acc_vpadalq((A_q) + 3 * (lda), _g, _blo,_bhi, &q30,&q31,&q32,&q33); \
        }                                                                     \
    } while (0)

/* 4-row × 8-col int32-out micro-kernel (A loaded once per row). */
static inline void qgemm_kernel_4x8_neon(int K,
                                         const int8_t *A_q, int lda,
                                         const int8_t *Bp,
                                         int32_t *C32, int ldc32) {
    int32x4_t q00=vdupq_n_s32(0),q01=vdupq_n_s32(0),q02=vdupq_n_s32(0),q03=vdupq_n_s32(0);
    int32x4_t q10=vdupq_n_s32(0),q11=vdupq_n_s32(0),q12=vdupq_n_s32(0),q13=vdupq_n_s32(0);
    int32x4_t q20=vdupq_n_s32(0),q21=vdupq_n_s32(0),q22=vdupq_n_s32(0),q23=vdupq_n_s32(0);
    int32x4_t q30=vdupq_n_s32(0),q31=vdupq_n_s32(0),q32=vdupq_n_s32(0),q33=vdupq_n_s32(0);
    FE_NEON_4x8_KLOOP(K, A_q, lda, Bp,
                      q00,q01,q02,q03, q10,q11,q12,q13,
                      q20,q21,q22,q23, q30,q31,q32,q33);
    vst1q_s32(C32 + 0*ldc32 + 0, vpaddq_s32(q00,q01)); vst1q_s32(C32 + 0*ldc32 + 4, vpaddq_s32(q02,q03));
    vst1q_s32(C32 + 1*ldc32 + 0, vpaddq_s32(q10,q11)); vst1q_s32(C32 + 1*ldc32 + 4, vpaddq_s32(q12,q13));
    vst1q_s32(C32 + 2*ldc32 + 0, vpaddq_s32(q20,q21)); vst1q_s32(C32 + 2*ldc32 + 4, vpaddq_s32(q22,q23));
    vst1q_s32(C32 + 3*ldc32 + 0, vpaddq_s32(q30,q31)); vst1q_s32(C32 + 3*ldc32 + 4, vpaddq_s32(q32,q33));
}

/* 8x8 int32-out wrapper: two 4x8 row-halves. Bit-identical to the
 * former 8x4-twice path; A LD1R traffic halved. */
static inline void qgemm_kernel_8x8_neon(int K,
                                         const int8_t *A_q, int lda,
                                         const int8_t *Bp,
                                         int32_t *C32, int ldc32) {
    qgemm_kernel_4x8_neon(K, A_q,             lda, Bp, C32,             ldc32);
    qgemm_kernel_4x8_neon(K, A_q + 4 * lda,   lda, Bp, C32 + 4 * ldc32, ldc32);
}

/* K=20 specialisation, vpadalq fused 8x4 (sub-block 0 or 1 of NR=8). The
 * k4 loop fully unrolls (5 groups). Caller invokes twice per NR=8 block. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x4_neon_k20(const int8_t *A_q, int lda,
                                              const int8_t *Bp,
                                              int32_t *C32, int ldc32) {
    int32x4_t p00 = vdupq_n_s32(0), p01 = vdupq_n_s32(0);
    int32x4_t p10 = vdupq_n_s32(0), p11 = vdupq_n_s32(0);
    int32x4_t p20 = vdupq_n_s32(0), p21 = vdupq_n_s32(0);
    int32x4_t p30 = vdupq_n_s32(0), p31 = vdupq_n_s32(0);
    int32x4_t p40 = vdupq_n_s32(0), p41 = vdupq_n_s32(0);
    int32x4_t p50 = vdupq_n_s32(0), p51 = vdupq_n_s32(0);
    int32x4_t p60 = vdupq_n_s32(0), p61 = vdupq_n_s32(0);
    int32x4_t p70 = vdupq_n_s32(0), p71 = vdupq_n_s32(0);

    /* K=20 → k4_groups=5 (odd). 2 vmlal2 pairs cover groups 0-1, 2-3,
     * then a single vpadalq tails group 4. */
    #define NEON_K20_VMLAL2_PAIR(G) do {                                      \
        int8x16_t b0 = vld1q_s8(Bp + (size_t)(G) * 32);                      \
        int8x16_t b1 = vld1q_s8(Bp + (size_t)((G) + 1) * 32);                \
        fe_neon_row_acc_vmlal2(0, (G), A_q, lda, b0, b1, &p00, &p01);        \
        fe_neon_row_acc_vmlal2(1, (G), A_q, lda, b0, b1, &p10, &p11);        \
        fe_neon_row_acc_vmlal2(2, (G), A_q, lda, b0, b1, &p20, &p21);        \
        fe_neon_row_acc_vmlal2(3, (G), A_q, lda, b0, b1, &p30, &p31);        \
        fe_neon_row_acc_vmlal2(4, (G), A_q, lda, b0, b1, &p40, &p41);        \
        fe_neon_row_acc_vmlal2(5, (G), A_q, lda, b0, b1, &p50, &p51);        \
        fe_neon_row_acc_vmlal2(6, (G), A_q, lda, b0, b1, &p60, &p61);        \
        fe_neon_row_acc_vmlal2(7, (G), A_q, lda, b0, b1, &p70, &p71);        \
    } while (0)
    #define NEON_K20_VPADAL_TAIL(G) do {                                      \
        int8x16_t b = vld1q_s8(Bp + (size_t)(G) * 32);                       \
        fe_neon_row_acc_vpadalq(0, (G), A_q, lda, b, &p00, &p01);            \
        fe_neon_row_acc_vpadalq(1, (G), A_q, lda, b, &p10, &p11);            \
        fe_neon_row_acc_vpadalq(2, (G), A_q, lda, b, &p20, &p21);            \
        fe_neon_row_acc_vpadalq(3, (G), A_q, lda, b, &p30, &p31);            \
        fe_neon_row_acc_vpadalq(4, (G), A_q, lda, b, &p40, &p41);            \
        fe_neon_row_acc_vpadalq(5, (G), A_q, lda, b, &p50, &p51);            \
        fe_neon_row_acc_vpadalq(6, (G), A_q, lda, b, &p60, &p61);            \
        fe_neon_row_acc_vpadalq(7, (G), A_q, lda, b, &p70, &p71);            \
    } while (0)
    NEON_K20_VMLAL2_PAIR(0);
    NEON_K20_VMLAL2_PAIR(2);
    NEON_K20_VPADAL_TAIL(4);
    #undef NEON_K20_VMLAL2_PAIR
    #undef NEON_K20_VPADAL_TAIL

    vst1q_s32(C32 + 0 * ldc32, vpaddq_s32(p00, p01));
    vst1q_s32(C32 + 1 * ldc32, vpaddq_s32(p10, p11));
    vst1q_s32(C32 + 2 * ldc32, vpaddq_s32(p20, p21));
    vst1q_s32(C32 + 3 * ldc32, vpaddq_s32(p30, p31));
    vst1q_s32(C32 + 4 * ldc32, vpaddq_s32(p40, p41));
    vst1q_s32(C32 + 5 * ldc32, vpaddq_s32(p50, p51));
    vst1q_s32(C32 + 6 * ldc32, vpaddq_s32(p60, p61));
    vst1q_s32(C32 + 7 * ldc32, vpaddq_s32(p70, p71));
}

/* 8x8 K=20 wrapper: 2× 8x4 fully-unrolled sub-blocks. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x8_neon_k20(const int8_t *A_q, int lda,
                                              const int8_t *Bp,
                                              int32_t *C32, int ldc32) {
    qgemm_kernel_8x4_neon_k20(A_q, lda, Bp +  0, C32 + 0, ldc32);
    qgemm_kernel_8x4_neon_k20(A_q, lda, Bp + 16, C32 + 4, ldc32);
}

void qgemm_neon_int32_k20(int M, int N,
                           const int8_t *A_q, const int8_t *Bp,
                           int32_t *C32, int ldc32) {
    enum { K = 20, MR = FE_QGEMM_MR, NR = FE_QGEMM_NR };
    enum { k4_groups = (K + 3) / 4 };
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_neon_k20(A_q + (size_t)mr * K, K,
                                       Bp + (size_t)bn * k4_groups * NR * 4,
                                       C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

void qgemm_neon_int32(int M, int N, int K,
                      const int8_t *A_q, const int8_t *Bp,
                      int32_t *C32, int ldc32) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_neon(K, A_q + (size_t)mr * K, K,
                                  Bp + (size_t)bn * k4_groups * NR * 4,
                                  C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* 4-col fused store: partial pair-reduce → fp32 dequant + bias + opt
 * SiLU + store. Single vpaddq combines the two i32x4 partials into a
 * 4-col i32x4 in [c0,c1,c2,c3] order. */
static inline void fe_neon_fused_store_row_4col(int R,
                                                 int32x4_t p_lo, int32x4_t p_hi,
                                                 float32x4_t vs,
                                                 const float *bias_n0,
                                                 int act_silu,
                                                 float *C, int ldc) {
    int32x4_t c = vpaddq_s32(p_lo, p_hi);
    float32x4_t v = vcvtq_f32_s32(c);
    if (bias_n0) v = vfmaq_f32(vld1q_f32(bias_n0), v, vs);
    else         v = vmulq_f32(v, vs);
    if (act_silu) v = vmulq_f32(v, FE_SIGMOIDF4(v));
    vst1q_f32(C + R * ldc, v);
}

/* 4-row × 8-col fp32-fused micro-kernel (A loaded once per row). */
static inline void qgemm_kernel_4x8_neon_fused(int K,
                                                const int8_t *A_q, int lda,
                                                const int8_t *Bp,
                                                const float *combined_scale_n0,
                                                const float *bias_n0,
                                                float *C, int ldc,
                                                int act_silu) {
    int32x4_t q00=vdupq_n_s32(0),q01=vdupq_n_s32(0),q02=vdupq_n_s32(0),q03=vdupq_n_s32(0);
    int32x4_t q10=vdupq_n_s32(0),q11=vdupq_n_s32(0),q12=vdupq_n_s32(0),q13=vdupq_n_s32(0);
    int32x4_t q20=vdupq_n_s32(0),q21=vdupq_n_s32(0),q22=vdupq_n_s32(0),q23=vdupq_n_s32(0);
    int32x4_t q30=vdupq_n_s32(0),q31=vdupq_n_s32(0),q32=vdupq_n_s32(0),q33=vdupq_n_s32(0);
    FE_NEON_4x8_KLOOP(K, A_q, lda, Bp,
                      q00,q01,q02,q03, q10,q11,q12,q13,
                      q20,q21,q22,q23, q30,q31,q32,q33);
    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    const float *bl = bias_n0 ? bias_n0 + 0 : NULL;
    const float *bh = bias_n0 ? bias_n0 + 4 : NULL;
    fe_neon_fused_store_row_4col(0, q00, q01, vs_lo, bl, act_silu, C,     ldc);
    fe_neon_fused_store_row_4col(0, q02, q03, vs_hi, bh, act_silu, C + 4, ldc);
    fe_neon_fused_store_row_4col(1, q10, q11, vs_lo, bl, act_silu, C,     ldc);
    fe_neon_fused_store_row_4col(1, q12, q13, vs_hi, bh, act_silu, C + 4, ldc);
    fe_neon_fused_store_row_4col(2, q20, q21, vs_lo, bl, act_silu, C,     ldc);
    fe_neon_fused_store_row_4col(2, q22, q23, vs_hi, bh, act_silu, C + 4, ldc);
    fe_neon_fused_store_row_4col(3, q30, q31, vs_lo, bl, act_silu, C,     ldc);
    fe_neon_fused_store_row_4col(3, q32, q33, vs_hi, bh, act_silu, C + 4, ldc);
}

/* 8x8 fp32-fused wrapper: two 4x8 row-halves (bit-identical). */
static inline void qgemm_kernel_8x8_neon_fused(int K,
                                                const int8_t *A_q, int lda,
                                                const int8_t *Bp,
                                                const float *combined_scale_n0,
                                                const float *bias_n0,
                                                float *C, int ldc,
                                                int act_silu) {
    qgemm_kernel_4x8_neon_fused(K, A_q,           lda, Bp, combined_scale_n0,
                                bias_n0, C,             ldc, act_silu);
    qgemm_kernel_4x8_neon_fused(K, A_q + 4 * lda, lda, Bp, combined_scale_n0,
                                bias_n0, C + 4 * ldc,   ldc, act_silu);
}


void qgemm_neon_fp32_fused(int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc,
                            int act_silu, int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_neon_fused(K, A_q + (size_t)mr * K, K,
                                         Bp + (size_t)bn * k4_groups * NR * 4,
                                         combined_scale + nr,
                                         bias ? bias + nr : NULL,
                                         C + (size_t)mr * ldc + nr, ldc,
                                         act_silu);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* 4-col fused-acc store: partial pair-reduce → fp32 dequant + bias →
 * accumulating add into C[][]. Matches the 2-pass FMA order:
 * tmp = bias + scale*c32 (vfmaq), then C += tmp (vaddq). */
static inline void fe_neon_fused_store_row_acc_4col(int R,
                                                     int32x4_t p_lo, int32x4_t p_hi,
                                                     float32x4_t vs,
                                                     const float *bias_n0,
                                                     float *C, int ldc) {
    int32x4_t c = vpaddq_s32(p_lo, p_hi);
    float32x4_t v = vcvtq_f32_s32(c);
    if (bias_n0) v = vfmaq_f32(vld1q_f32(bias_n0), v, vs);
    else         v = vmulq_f32(v, vs);
    float32x4_t cur = vld1q_f32(C + R * ldc);
    vst1q_f32(C + R * ldc, vaddq_f32(cur, v));
}

/* 4-row × 8-col fp32-fused-acc micro-kernel (A loaded once per row). */
static inline void qgemm_kernel_4x8_neon_fused_acc(int K,
                                                    const int8_t *A_q, int lda,
                                                    const int8_t *Bp,
                                                    const float *combined_scale_n0,
                                                    const float *bias_n0,
                                                    float *C, int ldc) {
    int32x4_t q00=vdupq_n_s32(0),q01=vdupq_n_s32(0),q02=vdupq_n_s32(0),q03=vdupq_n_s32(0);
    int32x4_t q10=vdupq_n_s32(0),q11=vdupq_n_s32(0),q12=vdupq_n_s32(0),q13=vdupq_n_s32(0);
    int32x4_t q20=vdupq_n_s32(0),q21=vdupq_n_s32(0),q22=vdupq_n_s32(0),q23=vdupq_n_s32(0);
    int32x4_t q30=vdupq_n_s32(0),q31=vdupq_n_s32(0),q32=vdupq_n_s32(0),q33=vdupq_n_s32(0);
    FE_NEON_4x8_KLOOP(K, A_q, lda, Bp,
                      q00,q01,q02,q03, q10,q11,q12,q13,
                      q20,q21,q22,q23, q30,q31,q32,q33);
    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    const float *bl = bias_n0 ? bias_n0 + 0 : NULL;
    const float *bh = bias_n0 ? bias_n0 + 4 : NULL;
    fe_neon_fused_store_row_acc_4col(0, q00, q01, vs_lo, bl, C,     ldc);
    fe_neon_fused_store_row_acc_4col(0, q02, q03, vs_hi, bh, C + 4, ldc);
    fe_neon_fused_store_row_acc_4col(1, q10, q11, vs_lo, bl, C,     ldc);
    fe_neon_fused_store_row_acc_4col(1, q12, q13, vs_hi, bh, C + 4, ldc);
    fe_neon_fused_store_row_acc_4col(2, q20, q21, vs_lo, bl, C,     ldc);
    fe_neon_fused_store_row_acc_4col(2, q22, q23, vs_hi, bh, C + 4, ldc);
    fe_neon_fused_store_row_acc_4col(3, q30, q31, vs_lo, bl, C,     ldc);
    fe_neon_fused_store_row_acc_4col(3, q32, q33, vs_hi, bh, C + 4, ldc);
}

/* 8x8 fp32-fused-acc wrapper: two 4x8 row-halves (bit-identical). */
static inline void qgemm_kernel_8x8_neon_fused_acc(int K,
                                                    const int8_t *A_q, int lda,
                                                    const int8_t *Bp,
                                                    const float *combined_scale_n0,
                                                    const float *bias_n0,
                                                    float *C, int ldc) {
    qgemm_kernel_4x8_neon_fused_acc(K, A_q,           lda, Bp, combined_scale_n0,
                                    bias_n0, C,           ldc);
    qgemm_kernel_4x8_neon_fused_acc(K, A_q + 4 * lda, lda, Bp, combined_scale_n0,
                                    bias_n0, C + 4 * ldc, ldc);
}


__attribute__((hot))
void qgemm_neon_fp32_fused_acc(int M, int N, int K,
                                const int8_t *A_q, const int8_t *Bp,
                                const float *combined_scale, const float *bias,
                                float *C, int ldc,
                                int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_neon_fused_acc(K, A_q + (size_t)mr * K, K,
                                             Bp + (size_t)bn * k4_groups * NR * 4,
                                             combined_scale + nr,
                                             bias ? bias + nr : NULL,
                                             C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* 4-col fused-track store: partial pair-reduce → dequant + bias →
 * store + accumulate max-abs into the caller's vmax. */
static inline void fe_neon_fused_store_row_track_4col(int R,
                                                       int32x4_t p_lo, int32x4_t p_hi,
                                                       float32x4_t vs,
                                                       const float *bias_n0,
                                                       float *C, int ldc,
                                                       float32x4_t *vmax) {
    int32x4_t c = vpaddq_s32(p_lo, p_hi);
    float32x4_t v = vcvtq_f32_s32(c);
    if (bias_n0) v = vfmaq_f32(vld1q_f32(bias_n0), v, vs);
    else         v = vmulq_f32(v, vs);
    vst1q_f32(C + R * ldc, v);
    *vmax = vmaxq_f32(*vmax, vabsq_f32(v));
}

static inline void qgemm_kernel_8x4_neon_fused_track(int K,
                                                      const int8_t *A_q, int lda,
                                                      const int8_t *Bp,
                                                      const float *combined_scale_n0,
                                                      const float *bias_n0,
                                                      float *C, int ldc,
                                                      float32x4_t *vmax) {
    int32x4_t p00 = vdupq_n_s32(0), p01 = vdupq_n_s32(0);
    int32x4_t p10 = vdupq_n_s32(0), p11 = vdupq_n_s32(0);
    int32x4_t p20 = vdupq_n_s32(0), p21 = vdupq_n_s32(0);
    int32x4_t p30 = vdupq_n_s32(0), p31 = vdupq_n_s32(0);
    int32x4_t p40 = vdupq_n_s32(0), p41 = vdupq_n_s32(0);
    int32x4_t p50 = vdupq_n_s32(0), p51 = vdupq_n_s32(0);
    int32x4_t p60 = vdupq_n_s32(0), p61 = vdupq_n_s32(0);
    int32x4_t p70 = vdupq_n_s32(0), p71 = vdupq_n_s32(0);

    int k4_groups = K / 4;
    int g = 0;
    for (; g + 1 < k4_groups; g += 2) {
        int8x16_t b0 = vld1q_s8(Bp + (size_t)g * 32);
        int8x16_t b1 = vld1q_s8(Bp + (size_t)(g + 1) * 32);
        fe_neon_row_acc_vmlal2(0, g, A_q, lda, b0, b1, &p00, &p01);
        fe_neon_row_acc_vmlal2(1, g, A_q, lda, b0, b1, &p10, &p11);
        fe_neon_row_acc_vmlal2(2, g, A_q, lda, b0, b1, &p20, &p21);
        fe_neon_row_acc_vmlal2(3, g, A_q, lda, b0, b1, &p30, &p31);
        fe_neon_row_acc_vmlal2(4, g, A_q, lda, b0, b1, &p40, &p41);
        fe_neon_row_acc_vmlal2(5, g, A_q, lda, b0, b1, &p50, &p51);
        fe_neon_row_acc_vmlal2(6, g, A_q, lda, b0, b1, &p60, &p61);
        fe_neon_row_acc_vmlal2(7, g, A_q, lda, b0, b1, &p70, &p71);
    }
    if (g < k4_groups) {
        int8x16_t b = vld1q_s8(Bp + (size_t)g * 32);
        fe_neon_row_acc_vpadalq(0, g, A_q, lda, b, &p00, &p01);
        fe_neon_row_acc_vpadalq(1, g, A_q, lda, b, &p10, &p11);
        fe_neon_row_acc_vpadalq(2, g, A_q, lda, b, &p20, &p21);
        fe_neon_row_acc_vpadalq(3, g, A_q, lda, b, &p30, &p31);
        fe_neon_row_acc_vpadalq(4, g, A_q, lda, b, &p40, &p41);
        fe_neon_row_acc_vpadalq(5, g, A_q, lda, b, &p50, &p51);
        fe_neon_row_acc_vpadalq(6, g, A_q, lda, b, &p60, &p61);
        fe_neon_row_acc_vpadalq(7, g, A_q, lda, b, &p70, &p71);
    }

    float32x4_t vs = vld1q_f32(combined_scale_n0);
    fe_neon_fused_store_row_track_4col(0, p00, p01, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(1, p10, p11, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(2, p20, p21, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(3, p30, p31, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(4, p40, p41, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(5, p50, p51, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(6, p60, p61, vs, bias_n0, C, ldc, vmax);
    fe_neon_fused_store_row_track_4col(7, p70, p71, vs, bias_n0, C, ldc, vmax);
}

/* 8x8 fp32-fused-track wrapper: 2× 8x4 calls, each updating its own vmax
 * register so the outer max-abs reduction keeps its 2-stream structure. */
static inline void qgemm_kernel_8x8_neon_fused_track(int K,
                                                      const int8_t *A_q, int lda,
                                                      const int8_t *Bp,
                                                      const float *combined_scale_n0,
                                                      const float *bias_n0,
                                                      float *C, int ldc,
                                                      float32x4_t *vmax_lo,
                                                      float32x4_t *vmax_hi) {
    qgemm_kernel_8x4_neon_fused_track(K, A_q, lda, Bp +  0,
                                       combined_scale_n0 + 0,
                                       bias_n0 ? bias_n0 + 0 : NULL,
                                       C + 0, ldc, vmax_lo);
    qgemm_kernel_8x4_neon_fused_track(K, A_q, lda, Bp + 16,
                                       combined_scale_n0 + 4,
                                       bias_n0 ? bias_n0 + 4 : NULL,
                                       C + 4, ldc, vmax_hi);
}

__attribute__((hot))
void qgemm_neon_fp32_fused_track_maxabs(int M, int N, int K,
                                         const int8_t *A_q, const int8_t *Bp,
                                         const float *combined_scale,
                                         const float *bias,
                                         float *C, int ldc,
                                         int32_t *c32_tail,
                                         float *max_abs_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    float32x4_t vmax_lo = vdupq_n_f32(0.0f);
    float32x4_t vmax_hi = vdupq_n_f32(0.0f);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_neon_fused_track(K, A_q + (size_t)mr * K, K,
                                               Bp + (size_t)bn * k4_groups * NR * 4,
                                               combined_scale + nr,
                                               bias ? bias + nr : NULL,
                                               C + (size_t)mr * ldc + nr, ldc,
                                               &vmax_lo, &vmax_hi);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
    float32x4_t vmx = vmaxq_f32(vmax_lo, vmax_hi);
    *max_abs_out = fe_hmax(vmx);
}

/*
 * qgemm_neon_gru_full_fused: fuse W_ih, W_hh and gate into one pass.
 * NEON-tier mirror of the DOTPROD/I8MM full-fused GRU kernels.
 */

static inline void fe_neon_dq_row_to_ih(int32x4_t acc_lo, int32x4_t acc_hi,
                                         float32x4_t vs_lo, float32x4_t vs_hi,
                                         float32x4_t vb_lo, float32x4_t vb_hi,
                                         float *ih_tile_row) {
    float32x4_t lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(acc_lo), vs_lo);
    float32x4_t hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(acc_hi), vs_hi);
    vst1q_f32(ih_tile_row + 0, lo);
    vst1q_f32(ih_tile_row + 4, hi);
}

static inline void fe_neon_rzgate_from_tile(int32x4_t acc_lo, int32x4_t acc_hi,
                                             float32x4_t vs_lo, float32x4_t vs_hi,
                                             float32x4_t vb_lo, float32x4_t vb_hi,
                                             const float *ih_tile_row,
                                             const float *bsum_n,
                                             float *band, int n_off, int ld_band,
                                             int r_idx) {
    float32x4_t hh_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(acc_lo), vs_lo);
    float32x4_t hh_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(acc_hi), vs_hi);
    float32x4_t ih_lo = vld1q_f32(ih_tile_row + 0);
    float32x4_t ih_hi = vld1q_f32(ih_tile_row + 4);
    float32x4_t pre_lo = vaddq_f32(vaddq_f32(ih_lo, hh_lo), vld1q_f32(bsum_n + 0));
    float32x4_t pre_hi = vaddq_f32(vaddq_f32(ih_hi, hh_hi), vld1q_f32(bsum_n + 4));
    float32x4_t g_lo = FE_SIGMOIDF4(pre_lo);
    float32x4_t g_hi = FE_SIGMOIDF4(pre_hi);
    vst1q_f32(band + r_idx * ld_band + n_off + 0, g_lo);
    vst1q_f32(band + r_idx * ld_band + n_off + 4, g_hi);
}

/* 4-row × 8-col helper for the GRU tile. Loads A once per row and
 * produces, per row r, lo_r = cols 0-3, hi_r = cols 4-7 — identical layout
 * and identical integer reduction to the legacy phase path. */
#define NEON_FULL_4X8(K, A_q, lda, Bp, lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3)       \
    do {                                                                      \
        int32x4_t _q00=vdupq_n_s32(0),_q01=vdupq_n_s32(0),_q02=vdupq_n_s32(0),_q03=vdupq_n_s32(0); \
        int32x4_t _q10=vdupq_n_s32(0),_q11=vdupq_n_s32(0),_q12=vdupq_n_s32(0),_q13=vdupq_n_s32(0); \
        int32x4_t _q20=vdupq_n_s32(0),_q21=vdupq_n_s32(0),_q22=vdupq_n_s32(0),_q23=vdupq_n_s32(0); \
        int32x4_t _q30=vdupq_n_s32(0),_q31=vdupq_n_s32(0),_q32=vdupq_n_s32(0),_q33=vdupq_n_s32(0); \
        FE_NEON_4x8_KLOOP((K),(A_q),(lda),(Bp),                              \
                          _q00,_q01,_q02,_q03, _q10,_q11,_q12,_q13,          \
                          _q20,_q21,_q22,_q23, _q30,_q31,_q32,_q33);          \
        lo0=vpaddq_s32(_q00,_q01); hi0=vpaddq_s32(_q02,_q03);                 \
        lo1=vpaddq_s32(_q10,_q11); hi1=vpaddq_s32(_q12,_q13);                 \
        lo2=vpaddq_s32(_q20,_q21); hi2=vpaddq_s32(_q22,_q23);                 \
        lo3=vpaddq_s32(_q30,_q31); hi3=vpaddq_s32(_q32,_q33);                 \
    } while (0)

#define NEON_FULL_8X8_TILE(K, A_q, lda, Bp,                                  \
                           c00,c01,c10,c11,c20,c21,c30,c31,                  \
                           c40,c41,c50,c51,c60,c61,c70,c71)                  \
    do {                                                                      \
        NEON_FULL_4X8((K), (A_q),             (lda), (Bp),                    \
                      c00,c01, c10,c11, c20,c21, c30,c31);                    \
        NEON_FULL_4X8((K), (A_q) + 4 * (lda), (lda), (Bp),                    \
                      c40,c41, c50,c51, c60,c61, c70,c71);                    \
    } while (0)

/* NEON fp16-inout variant: gru_h stored as IEEE fp16.
 * h_old loads from fp16 storage via FCVTL, h_new dual-stores (fp32
 * scratch for rnn_fc + fp16 storage update). */
static inline void fe_neon_ngate_from_tile_fp16inout(
        int32x4_t acc_lo, int32x4_t acc_hi,
        float32x4_t vs_lo, float32x4_t vs_hi,
        float32x4_t vb_lo, float32x4_t vb_hi,
        const float *ih_tile_row,
        const float *bn_i_n, const float *bn_h_n,
        const float *r_band, const float *z_band,
        int n_off, int ld_band,
        uint16_t *h_row_fp16,
        float *h_out_row, int r_idx) {
    float32x4_t hh_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(acc_lo), vs_lo);
    float32x4_t hh_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(acc_hi), vs_hi);
    float32x4_t bi_lo = vld1q_f32(bn_i_n + 0), bi_hi = vld1q_f32(bn_i_n + 4);
    float32x4_t bh_lo = vld1q_f32(bn_h_n + 0), bh_hi = vld1q_f32(bn_h_n + 4);
    float32x4_t ih_lo = vld1q_f32(ih_tile_row + 0);
    float32x4_t ih_hi = vld1q_f32(ih_tile_row + 4);
    float32x4_t ihp_lo = vaddq_f32(ih_lo, bi_lo);
    float32x4_t ihp_hi = vaddq_f32(ih_hi, bi_hi);
    float32x4_t hhp_lo = vaddq_f32(hh_lo, bh_lo);
    float32x4_t hhp_hi = vaddq_f32(hh_hi, bh_hi);
    float32x4_t rb_lo = vld1q_f32(r_band + r_idx * ld_band + n_off + 0);
    float32x4_t rb_hi = vld1q_f32(r_band + r_idx * ld_band + n_off + 4);
    float32x4_t np_lo = vfmaq_f32(ihp_lo, rb_lo, hhp_lo);
    float32x4_t np_hi = vfmaq_f32(ihp_hi, rb_hi, hhp_hi);
    float32x4_t n_lo = FE_TANHF4(np_lo);
    float32x4_t n_hi = FE_TANHF4(np_hi);
    float32x4_t zb_lo = vld1q_f32(z_band + r_idx * ld_band + n_off + 0);
    float32x4_t zb_hi = vld1q_f32(z_band + r_idx * ld_band + n_off + 4);
    float32x4_t ho_lo = fe_fp16_load4(h_row_fp16 + n_off + 0);
    float32x4_t ho_hi = fe_fp16_load4(h_row_fp16 + n_off + 4);
    float32x4_t hn_lo = vfmaq_f32(n_lo, zb_lo, vsubq_f32(ho_lo, n_lo));
    float32x4_t hn_hi = vfmaq_f32(n_hi, zb_hi, vsubq_f32(ho_hi, n_hi));
    vst1q_f32(h_out_row + n_off + 0, hn_lo);
    vst1q_f32(h_out_row + n_off + 4, hn_hi);
    fe_fp16_store4(h_row_fp16 + n_off + 0, hn_lo);
    fe_fp16_store4(h_row_fp16 + n_off + 4, hn_hi);
}

__attribute__((hot))
void qgemm_neon_gru_full_fused_fp16inout(int M, int D,
                                          const int8_t *Xq, int ld_x,
                                          const int8_t *Hq, int ld_h,
                                          const int8_t *Wq_ih,
                                          const int8_t *Wq_hh,
                                          const float *combined_ih,
                                          const float *bias_eff_ih,
                                          const float *combined_hh,
                                          const float *bias_eff_hh,
                                          const float *br_sum, const float *bz_sum,
                                          const float *bn_i,   const float *bn_h,
                                          uint16_t *h_inout_fp16, int ld_h_inout,
                                          float *h_out_scratch, int ld_h_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    const int K = D;
    const int k4_groups = (K + 3) / 4;
    const size_t weight_n_block_stride = (size_t)k4_groups * NR * 4;
    const int N_blocks_per_gate = D / NR;

    enum { MAX_MR = FE_QGEMM_MR };
    const int ld_band = D;
    float *r_band = fe_gru_r_band;
    float *z_band = fe_gru_z_band;
    float ih_tile[MAX_MR * NR];

    int32x4_t c00, c01, c10, c11, c20, c21, c30, c31;
    int32x4_t c40, c41, c50, c51, c60, c61, c70, c71;

    for (int mr = 0; mr + MR <= M; mr += MR) {
        const int8_t  *X_block          = Xq + (size_t)mr * ld_x;
        const int8_t  *H_block          = Hq + (size_t)mr * ld_h;
        uint16_t      *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float         *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;

        /* r gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)nb * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)nb * weight_n_block_stride;
            int n_off = nb * NR;
            NEON_FULL_8X8_TILE(K, X_block, ld_x, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + n_off + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + n_off + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + n_off + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + n_off + 4);
            fe_neon_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_neon_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_neon_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_neon_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_neon_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_neon_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_neon_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_neon_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            NEON_FULL_8X8_TILE(K, H_block, ld_h, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + n_off + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + n_off + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + n_off + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + n_off + 4);
            fe_neon_rzgate_from_tile(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, br_sum + n_off, r_band, n_off, ld_band, 0);
            fe_neon_rzgate_from_tile(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, br_sum + n_off, r_band, n_off, ld_band, 1);
            fe_neon_rzgate_from_tile(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, br_sum + n_off, r_band, n_off, ld_band, 2);
            fe_neon_rzgate_from_tile(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, br_sum + n_off, r_band, n_off, ld_band, 3);
            fe_neon_rzgate_from_tile(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, br_sum + n_off, r_band, n_off, ld_band, 4);
            fe_neon_rzgate_from_tile(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, br_sum + n_off, r_band, n_off, ld_band, 5);
            fe_neon_rzgate_from_tile(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, br_sum + n_off, r_band, n_off, ld_band, 6);
            fe_neon_rzgate_from_tile(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, br_sum + n_off, r_band, n_off, ld_band, 7);
        }

        /* z gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = D + n_off;
            NEON_FULL_8X8_TILE(K, X_block, ld_x, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_neon_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_neon_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_neon_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_neon_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_neon_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_neon_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_neon_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_neon_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            NEON_FULL_8X8_TILE(K, H_block, ld_h, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_neon_rzgate_from_tile(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, bz_sum + n_off, z_band, n_off, ld_band, 0);
            fe_neon_rzgate_from_tile(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, bz_sum + n_off, z_band, n_off, ld_band, 1);
            fe_neon_rzgate_from_tile(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, bz_sum + n_off, z_band, n_off, ld_band, 2);
            fe_neon_rzgate_from_tile(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, bz_sum + n_off, z_band, n_off, ld_band, 3);
            fe_neon_rzgate_from_tile(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, bz_sum + n_off, z_band, n_off, ld_band, 4);
            fe_neon_rzgate_from_tile(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, bz_sum + n_off, z_band, n_off, ld_band, 5);
            fe_neon_rzgate_from_tile(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, bz_sum + n_off, z_band, n_off, ld_band, 6);
            fe_neon_rzgate_from_tile(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, bz_sum + n_off, z_band, n_off, ld_band, 7);
        }

        /* n gate + h_new (fp16 dual store) */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = 2 * D + n_off;
            NEON_FULL_8X8_TILE(K, X_block, ld_x, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_neon_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_neon_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_neon_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_neon_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_neon_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_neon_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_neon_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_neon_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            NEON_FULL_8X8_TILE(K, H_block, ld_h, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_neon_ngate_from_tile_fp16inout(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 0 * ld_h_inout, h_block_scratch + 0 * ld_h_out, 0);
            fe_neon_ngate_from_tile_fp16inout(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 1 * ld_h_inout, h_block_scratch + 1 * ld_h_out, 1);
            fe_neon_ngate_from_tile_fp16inout(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 2 * ld_h_inout, h_block_scratch + 2 * ld_h_out, 2);
            fe_neon_ngate_from_tile_fp16inout(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 3 * ld_h_inout, h_block_scratch + 3 * ld_h_out, 3);
            fe_neon_ngate_from_tile_fp16inout(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 4 * ld_h_inout, h_block_scratch + 4 * ld_h_out, 4);
            fe_neon_ngate_from_tile_fp16inout(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 5 * ld_h_inout, h_block_scratch + 5 * ld_h_out, 5);
            fe_neon_ngate_from_tile_fp16inout(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 6 * ld_h_inout, h_block_scratch + 6 * ld_h_out, 6);
            fe_neon_ngate_from_tile_fp16inout(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 7 * ld_h_inout, h_block_scratch + 7 * ld_h_out, 7);
        }
    }
}

#undef NEON_FULL_8X8_TILE

#endif /* aarch64 */

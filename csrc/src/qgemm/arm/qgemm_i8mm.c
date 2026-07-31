/*
 * arm/qgemm_i8mm.c - ARMv8.6+ FEAT_I8MM (I8MM) int8 GEMM tier.
 *
 * 8x8 microtile uses 16 I8MMs per k8-group. Requires K multiple of 8;
 * otherwise falls back to the DOTPROD tier. Fused epilogues match the
 * 2-pass dequant FMA order for bit-identical results.
 */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

#define FE_QPOST_NEON 1
#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../../internal/fe_simd.h"
#include "../../internal/fe_fp16.h"  /* fp16 inout: load/store helpers */
#include "../qgemm_arch.h"
#include "../qgemm_simd_post.inl"
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* A row-pair pre-pack scratch buffers (KleidiAI-style layout).
 *
 * SMMLA's `a` operand is 16 bytes = [r0_K0..K7 | r1_K0..K7] (2 rows × 8 K).
 * The legacy `fe_i8mm_k8_acc` builds this on the fly via 8 vld1_s8 + 4
 * vcombine = 12 µops per K-block. With A pre-packed as the same layout
 * (4 row-pairs × 16 B per K-block = 64 bytes contiguous), a single
 * `vld1q_s8_x4` covers all 4 row-pairs = 1 µop per K-block. Saves
 * 8 µops/K-block, which on Apple M2's 4-wide dispatch pulls the
 * front-end well off the SMMLA dispatch boundary.
 *
 * Per MR=8 tile the packed bytes = 4 row-pairs × k8_groups × 16 = 8*K B.
 * Total per call = (M/MR)*MR*K = M*K bytes (same as original A).
 *
 * Worst Medium runtime M*K = 128 (F1) * 288 (Conv k=3) = 36 KiB.
 * Static scratch bound = 256 * FE_QGEMM_MAX_K = 98,304 B.
 * GRU has two simultaneous inputs (X and H), each at most D*D bytes;
 * dedicated buffers avoid the overlap.
 */
#define FE_I8MM_MAX_A_MK   (256 * FE_QGEMM_MAX_K)
#define FE_I8MM_GRU_MAX_MK (FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D)

static int8_t fe_i8mm_a_rp_buf[FE_I8MM_MAX_A_MK];
static int8_t fe_i8mm_gru_x_rp_buf[FE_I8MM_GRU_MAX_MK];
static int8_t fe_i8mm_gru_h_rp_buf[FE_I8MM_GRU_MAX_MK];

/* Peak-stabilization: pre-fault the row-pair BSS scratch pages
 * at fe_qgemm_init (the ARM mirror of the AVX2 tier's BSS pre-fault).
 * Anon BSS is COW-zero on macOS/Linux/Android, so first-touch causes a
 * minor page fault that can leak into steady-state percentiles. Apple
 * Silicon uses 16 KB pages; stride at 4 KB safely commits every 16 KB page
 * (4× redundant writes, negligible at init). The three buffers total 128 KiB
 * (32 nominal 4 KB pages). Called from fe_qgemm_init on aarch64 regardless
 * of selected tier -- the row-pair buffers are tier-static and the cost is
 * outside steady-state timing. */
void qgemm_arm_i8mm_prefault_buffers(void) {
    const size_t PAGE = 4096;
    volatile unsigned char *p;

    p = (volatile unsigned char *)fe_i8mm_a_rp_buf;
    for (size_t i = 0; i < sizeof(fe_i8mm_a_rp_buf); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_i8mm_gru_x_rp_buf;
    for (size_t i = 0; i < sizeof(fe_i8mm_gru_x_rp_buf); i += PAGE) p[i] = 0;

    p = (volatile unsigned char *)fe_i8mm_gru_h_rp_buf;
    for (size_t i = 0; i < sizeof(fe_i8mm_gru_h_rp_buf); i += PAGE) p[i] = 0;
}

/* Repack A from row-major [M, K] (stride lda) into the row-pair layout.
 * Only full MR=8 tiles are written. M%MR and N%NR are always 0 (tile-aligned
 * dims, enforced by the _Static_assert block in fe_config_medium.h), so the
 * remainder is unreachable; a non-tile dim would hit fe_qgemm_tail_unsupported. */
static inline void fe_i8mm_repack_a_rp(const int8_t *A_q, int M, int K, int lda,
                                        int8_t *A_rp) {
    int n_full_tiles = M / FE_QGEMM_MR;
    int k8_groups = K / 8;
    for (int t = 0; t < n_full_tiles; ++t) {
        int row0 = t * FE_QGEMM_MR;
        int8_t *tile_out = A_rp + (size_t)t * FE_QGEMM_MR * K;
        for (int g8 = 0; g8 < k8_groups; ++g8) {
            int8_t *bk = tile_out + (size_t)g8 * 64;
            for (int rp = 0; rp < 4; ++rp) {
                int rA = row0 + 2 * rp;
                int rB = row0 + 2 * rp + 1;
                int8x8_t va = vld1_s8(A_q + (size_t)rA * lda + (size_t)g8 * 8);
                int8x8_t vb = vld1_s8(A_q + (size_t)rB * lda + (size_t)g8 * 8);
                vst1q_s8(bk + (size_t)rp * 16, vcombine_s8(va, vb));
            }
        }
    }
}

/* New k8 accumulator that reads row-pair-packed A directly.
 * A_rp_tile points to the start of this MR tile's packed A (8*K bytes);
 * `g8 * 64` advances to the next K-block (4 row-pairs × 16 B). */
static inline void fe_i8mm_k8_acc_rp(const int8_t *A_rp_tile, int g8,
                                      int8x16_t b0, int8x16_t b1,
                                      int8x16_t b2, int8x16_t b3,
                                      int32x4_t *c00, int32x4_t *c01,
                                      int32x4_t *c02, int32x4_t *c03,
                                      int32x4_t *c10, int32x4_t *c11,
                                      int32x4_t *c12, int32x4_t *c13,
                                      int32x4_t *c20, int32x4_t *c21,
                                      int32x4_t *c22, int32x4_t *c23,
                                      int32x4_t *c30, int32x4_t *c31,
                                      int32x4_t *c32, int32x4_t *c33) {
    int8x16x4_t _a = vld1q_s8_x4(A_rp_tile + (size_t)g8 * 64);
    int8x16_t a0 = _a.val[0];  /* row-pair 0 = rows 0,1 */
    int8x16_t a1 = _a.val[1];  /* row-pair 1 = rows 2,3 */
    int8x16_t a2 = _a.val[2];  /* row-pair 2 = rows 4,5 */
    int8x16_t a3 = _a.val[3];  /* row-pair 3 = rows 6,7 */
    *c00 = vmmlaq_s32(*c00, a0, b0); *c01 = vmmlaq_s32(*c01, a0, b1);
    *c02 = vmmlaq_s32(*c02, a0, b2); *c03 = vmmlaq_s32(*c03, a0, b3);
    *c10 = vmmlaq_s32(*c10, a1, b0); *c11 = vmmlaq_s32(*c11, a1, b1);
    *c12 = vmmlaq_s32(*c12, a1, b2); *c13 = vmmlaq_s32(*c13, a1, b3);
    *c20 = vmmlaq_s32(*c20, a2, b0); *c21 = vmmlaq_s32(*c21, a2, b1);
    *c22 = vmmlaq_s32(*c22, a2, b2); *c23 = vmmlaq_s32(*c23, a2, b3);
    *c30 = vmmlaq_s32(*c30, a3, b0); *c31 = vmmlaq_s32(*c31, a3, b1);
    *c32 = vmmlaq_s32(*c32, a3, b2); *c33 = vmmlaq_s32(*c33, a3, b3);
}

/* The old row-major A-load k8 accumulator was removed -- all 5 I8MM kernel
 * variants now consume row-pair pre-packed A via fe_i8mm_k8_acc_rp.
 * The M%MR / N%NR remainder paths are unreachable (tile-aligned dims,
 * enforced in fe_config_medium.h) and call fe_qgemm_tail_unsupported(). */

/* SMMLA (vmmlaq_s32) writes a 2x2 i32 output tile per accumulator: the two
 * result ROWS land interleaved across the low/high 64-bit halves -- ci holds
 * {row0[2k], row0[2k+1], row1[2k], row1[2k+1]} for a column-pair k, i.e. the
 * lo 64b is row0's two cols and the hi 64b is row1's two cols. vtrn1q_s64 /
 * vtrn2q_s64 de-interleave those 64-bit halves: trn1 gathers all the lo
 * halves (-> row0) and trn2 all the hi halves (-> row1), restoring row-major
 * order so the epilogue can vst1q contiguous rows. This 64-bit de-interleave
 * is the least-obvious step in the I8MM tier. */
static inline void fe_i8mm_gather_rowpair(int32x4_t c0, int32x4_t c1,
                                           int32x4_t c2, int32x4_t c3,
                                           int32x4_t *row0_lo, int32x4_t *row0_hi,
                                           int32x4_t *row1_lo, int32x4_t *row1_hi) {
    *row0_lo = vreinterpretq_s32_s64(
        vtrn1q_s64(vreinterpretq_s64_s32(c0), vreinterpretq_s64_s32(c1)));
    *row0_hi = vreinterpretq_s32_s64(
        vtrn1q_s64(vreinterpretq_s64_s32(c2), vreinterpretq_s64_s32(c3)));
    *row1_lo = vreinterpretq_s32_s64(
        vtrn2q_s64(vreinterpretq_s64_s32(c0), vreinterpretq_s64_s32(c1)));
    *row1_hi = vreinterpretq_s32_s64(
        vtrn2q_s64(vreinterpretq_s64_s32(c2), vreinterpretq_s64_s32(c3)));
}

/* int32-out */
static inline void fe_i8mm_write_rowpair_int32(int rp,
                                                int32x4_t c0, int32x4_t c1,
                                                int32x4_t c2, int32x4_t c3,
                                                int32_t *C32, int ldc32) {
    int32x4_t r0_lo, r0_hi, r1_lo, r1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &r0_lo, &r0_hi, &r1_lo, &r1_hi);
    vst1q_s32(C32 + (2 * rp + 0) * ldc32 + 0, r0_lo);
    vst1q_s32(C32 + (2 * rp + 0) * ldc32 + 4, r0_hi);
    vst1q_s32(C32 + (2 * rp + 1) * ldc32 + 0, r1_lo);
    vst1q_s32(C32 + (2 * rp + 1) * ldc32 + 4, r1_hi);
}

static inline void qgemm_kernel_8x8_i8mm_rp(int k8_groups,
                                              const int8_t *A_rp_tile,
                                              const int8_t *Bp,
                                              int32_t *C32, int ldc32) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0),
              c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0),
              c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0),
              c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0),
              c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    for (int g8 = 0; g8 < k8_groups; ++g8) {
        int8x16x4_t _bp = vld1q_s8_x4(Bp + g8 * 64);
        int8x16_t b0 = _bp.val[0];
        int8x16_t b1 = _bp.val[1];
        int8x16_t b2 = _bp.val[2];
        int8x16_t b3 = _bp.val[3];
        fe_i8mm_k8_acc_rp(A_rp_tile, g8, b0, b1, b2, b3,
                          &c00, &c01, &c02, &c03,
                          &c10, &c11, &c12, &c13,
                          &c20, &c21, &c22, &c23,
                          &c30, &c31, &c32, &c33);
    }

    fe_i8mm_write_rowpair_int32(0, c00, c01, c02, c03, C32, ldc32);
    fe_i8mm_write_rowpair_int32(1, c10, c11, c12, c13, C32, ldc32);
    fe_i8mm_write_rowpair_int32(2, c20, c21, c22, c23, C32, ldc32);
    fe_i8mm_write_rowpair_int32(3, c30, c31, c32, c33, C32, ldc32);
}

void qgemm_i8mm_int32(int M, int N, int K,
                       const int8_t *A_q, const int8_t *Bp,
                       int32_t *C32, int ldc32) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    if ((K & 7) != 0) {
        qgemm_dotprod_int32(M, N, K, A_q, Bp, C32, ldc32);
        return;
    }
    /* one-shot row-pair repack of A; reused across all (mr, nr) tiles. */
    int k8_groups = K / 8;
    fe_i8mm_repack_a_rp(A_q, M, K, K, fe_i8mm_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_i8mm_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_i8mm_rp(k8_groups, A_rp_tile,
                                   Bp + (size_t)bn * k4_groups * NR * 4,
                                   C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* fp32-out fused */
static inline void fe_i8mm_fused_write_rowpair(int rp,
                                                int32x4_t c0, int32x4_t c1,
                                                int32x4_t c2, int32x4_t c3,
                                                float32x4_t vs_lo, float32x4_t vs_hi,
                                                const float *bias_n0,
                                                int act_silu,
                                                float *C, int ldc) {
    int32x4_t r0_lo, r0_hi, r1_lo, r1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &r0_lo, &r0_hi, &r1_lo, &r1_hi);
    float32x4_t vl0 = vcvtq_f32_s32(r0_lo);
    float32x4_t vh0 = vcvtq_f32_s32(r0_hi);
    float32x4_t vl1 = vcvtq_f32_s32(r1_lo);
    float32x4_t vh1 = vcvtq_f32_s32(r1_hi);
    if (bias_n0) {
        float32x4_t vb_lo = vld1q_f32(bias_n0 + 0);
        float32x4_t vb_hi = vld1q_f32(bias_n0 + 4);
        vl0 = vfmaq_f32(vb_lo, vl0, vs_lo);
        vh0 = vfmaq_f32(vb_hi, vh0, vs_hi);
        vl1 = vfmaq_f32(vb_lo, vl1, vs_lo);
        vh1 = vfmaq_f32(vb_hi, vh1, vs_hi);
    } else {
        vl0 = vmulq_f32(vl0, vs_lo); vh0 = vmulq_f32(vh0, vs_hi);
        vl1 = vmulq_f32(vl1, vs_lo); vh1 = vmulq_f32(vh1, vs_hi);
    }
    if (act_silu) {
        vl0 = fe_mul(vl0, FE_SIGMOIDF4(vl0));
        vh0 = fe_mul(vh0, FE_SIGMOIDF4(vh0));
        vl1 = fe_mul(vl1, FE_SIGMOIDF4(vl1));
        vh1 = fe_mul(vh1, FE_SIGMOIDF4(vh1));
    }
    vst1q_f32(C + (2 * rp + 0) * ldc + 0, vl0);
    vst1q_f32(C + (2 * rp + 0) * ldc + 4, vh0);
    vst1q_f32(C + (2 * rp + 1) * ldc + 0, vl1);
    vst1q_f32(C + (2 * rp + 1) * ldc + 4, vh1);
}

static inline void qgemm_kernel_8x8_i8mm_fused_rp(int k8_groups,
                                                const int8_t *A_rp_tile,
                                                const int8_t *Bp,
                                                const float *combined_scale_n0,
                                                const float *bias_n0,
                                                float *C, int ldc,
                                                int act_silu) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0),
              c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0),
              c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0),
              c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0),
              c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    for (int g8 = 0; g8 < k8_groups; ++g8) {
        int8x16x4_t _bp = vld1q_s8_x4(Bp + g8 * 64);
        int8x16_t b0 = _bp.val[0];
        int8x16_t b1 = _bp.val[1];
        int8x16_t b2 = _bp.val[2];
        int8x16_t b3 = _bp.val[3];
        fe_i8mm_k8_acc_rp(A_rp_tile, g8, b0, b1, b2, b3,
                          &c00, &c01, &c02, &c03,
                          &c10, &c11, &c12, &c13,
                          &c20, &c21, &c22, &c23,
                          &c30, &c31, &c32, &c33);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_i8mm_fused_write_rowpair(0, c00, c01, c02, c03, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_i8mm_fused_write_rowpair(1, c10, c11, c12, c13, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_i8mm_fused_write_rowpair(2, c20, c21, c22, c23, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_i8mm_fused_write_rowpair(3, c30, c31, c32, c33, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
}


void qgemm_i8mm_fp32_fused(int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc,
                            int act_silu, int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    if ((K & 7) != 0) {
        qgemm_dotprod_fp32_fused(M, N, K, A_q, Bp, combined_scale, bias, C, ldc,
                              act_silu, c32_tail);
        return;
    }
    int k8_groups = K / 8;
    fe_i8mm_repack_a_rp(A_q, M, K, K, fe_i8mm_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_i8mm_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_i8mm_fused_rp(
                k8_groups, A_rp_tile,
                Bp + (size_t)bn * k4_groups * NR * 4,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* fp32-out fused, accumulating epilogue (C += FC). */
static inline void fe_i8mm_fused_write_rowpair_acc(
        int rp,
        int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3,
        float32x4_t vs_lo, float32x4_t vs_hi,
        const float *bias_n0,
        float *C, int ldc) {
    int32x4_t r0_lo, r0_hi, r1_lo, r1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &r0_lo, &r0_hi, &r1_lo, &r1_hi);
    float32x4_t vl0 = vcvtq_f32_s32(r0_lo);
    float32x4_t vh0 = vcvtq_f32_s32(r0_hi);
    float32x4_t vl1 = vcvtq_f32_s32(r1_lo);
    float32x4_t vh1 = vcvtq_f32_s32(r1_hi);
    /* tmp = bias + c32 * scale */
    if (bias_n0) {
        float32x4_t vb_lo = vld1q_f32(bias_n0 + 0);
        float32x4_t vb_hi = vld1q_f32(bias_n0 + 4);
        vl0 = vfmaq_f32(vb_lo, vl0, vs_lo);
        vh0 = vfmaq_f32(vb_hi, vh0, vs_hi);
        vl1 = vfmaq_f32(vb_lo, vl1, vs_lo);
        vh1 = vfmaq_f32(vb_hi, vh1, vs_hi);
    } else {
        vl0 = vmulq_f32(vl0, vs_lo); vh0 = vmulq_f32(vh0, vs_hi);
        vl1 = vmulq_f32(vl1, vs_lo); vh1 = vmulq_f32(vh1, vs_hi);
    }
    /* C += tmp */
    vl0 = vaddq_f32(vl0, vld1q_f32(C + (2 * rp + 0) * ldc + 0));
    vh0 = vaddq_f32(vh0, vld1q_f32(C + (2 * rp + 0) * ldc + 4));
    vl1 = vaddq_f32(vl1, vld1q_f32(C + (2 * rp + 1) * ldc + 0));
    vh1 = vaddq_f32(vh1, vld1q_f32(C + (2 * rp + 1) * ldc + 4));
    vst1q_f32(C + (2 * rp + 0) * ldc + 0, vl0);
    vst1q_f32(C + (2 * rp + 0) * ldc + 4, vh0);
    vst1q_f32(C + (2 * rp + 1) * ldc + 0, vl1);
    vst1q_f32(C + (2 * rp + 1) * ldc + 4, vh1);
}

static inline void qgemm_kernel_8x8_i8mm_fused_acc_rp(
        int k8_groups,
        const int8_t *A_rp_tile,
        const int8_t *Bp,
        const float *combined_scale_n0,
        const float *bias_n0,
        float *C, int ldc) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0),
              c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0),
              c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0),
              c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0),
              c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    for (int g8 = 0; g8 < k8_groups; ++g8) {
        int8x16x4_t _bp = vld1q_s8_x4(Bp + g8 * 64);
        int8x16_t b0 = _bp.val[0];
        int8x16_t b1 = _bp.val[1];
        int8x16_t b2 = _bp.val[2];
        int8x16_t b3 = _bp.val[3];
        fe_i8mm_k8_acc_rp(A_rp_tile, g8, b0, b1, b2, b3,
                          &c00, &c01, &c02, &c03,
                          &c10, &c11, &c12, &c13,
                          &c20, &c21, &c22, &c23,
                          &c30, &c31, &c32, &c33);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_i8mm_fused_write_rowpair_acc(0, c00, c01, c02, c03, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_i8mm_fused_write_rowpair_acc(1, c10, c11, c12, c13, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_i8mm_fused_write_rowpair_acc(2, c20, c21, c22, c23, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_i8mm_fused_write_rowpair_acc(3, c30, c31, c32, c33, vs_lo, vs_hi, bias_n0, C, ldc);
}


__attribute__((hot))
void qgemm_i8mm_fp32_fused_acc(int M, int N, int K,
                                 const int8_t *A_q, const int8_t *Bp,
                                 const float *combined_scale, const float *bias,
                                 float *C, int ldc,
                                 int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    if ((K & 7) != 0) {
        /* No DOTPROD acc variant; the dispatch gates this on tier==I8MM so the
         * fallback branch is unreachable in practice. Leaves C overwritten. */
        qgemm_dotprod_fp32_fused(M, N, K, A_q, Bp, combined_scale, bias,
                              C, ldc, 0, c32_tail);
        return;
    }
    int k8_groups = K / 8;
    fe_i8mm_repack_a_rp(A_q, M, K, K, fe_i8mm_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_i8mm_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_i8mm_fused_acc_rp(
                k8_groups, A_rp_tile,
                Bp + (size_t)bn * k4_groups * NR * 4,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/*
 * fp32-out fused with max-abs tracking. Caller (MHSA QKV projection)
 * horizontally-reduces vmax_lo/vmax_hi after the kernel finishes.
 * No SiLU branch.
 */
static inline void fe_i8mm_fused_write_rowpair_track(
        int rp,
        int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3,
        float32x4_t vs_lo, float32x4_t vs_hi,
        const float *bias_n0,
        float *C, int ldc,
        float32x4_t *vmax_lo, float32x4_t *vmax_hi) {
    int32x4_t r0_lo, r0_hi, r1_lo, r1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &r0_lo, &r0_hi, &r1_lo, &r1_hi);
    float32x4_t vl0 = vcvtq_f32_s32(r0_lo);
    float32x4_t vh0 = vcvtq_f32_s32(r0_hi);
    float32x4_t vl1 = vcvtq_f32_s32(r1_lo);
    float32x4_t vh1 = vcvtq_f32_s32(r1_hi);
    if (bias_n0) {
        float32x4_t vb_lo = vld1q_f32(bias_n0 + 0);
        float32x4_t vb_hi = vld1q_f32(bias_n0 + 4);
        vl0 = vfmaq_f32(vb_lo, vl0, vs_lo);
        vh0 = vfmaq_f32(vb_hi, vh0, vs_hi);
        vl1 = vfmaq_f32(vb_lo, vl1, vs_lo);
        vh1 = vfmaq_f32(vb_hi, vh1, vs_hi);
    } else {
        vl0 = vmulq_f32(vl0, vs_lo); vh0 = vmulq_f32(vh0, vs_hi);
        vl1 = vmulq_f32(vl1, vs_lo); vh1 = vmulq_f32(vh1, vs_hi);
    }
    vst1q_f32(C + (2 * rp + 0) * ldc + 0, vl0);
    vst1q_f32(C + (2 * rp + 0) * ldc + 4, vh0);
    vst1q_f32(C + (2 * rp + 1) * ldc + 0, vl1);
    vst1q_f32(C + (2 * rp + 1) * ldc + 4, vh1);
    *vmax_lo = vmaxq_f32(*vmax_lo,
                          vmaxq_f32(vabsq_f32(vl0), vabsq_f32(vl1)));
    *vmax_hi = vmaxq_f32(*vmax_hi,
                          vmaxq_f32(vabsq_f32(vh0), vabsq_f32(vh1)));
}

static inline void qgemm_kernel_8x8_i8mm_fused_track_rp(
        int k8_groups,
        const int8_t *A_rp_tile,
        const int8_t *Bp,
        const float *combined_scale_n0,
        const float *bias_n0,
        float *C, int ldc,
        float32x4_t *vmax_lo, float32x4_t *vmax_hi) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0),
              c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0),
              c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0),
              c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0),
              c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);

    for (int g8 = 0; g8 < k8_groups; ++g8) {
        int8x16x4_t _bp = vld1q_s8_x4(Bp + g8 * 64);
        int8x16_t b0 = _bp.val[0];
        int8x16_t b1 = _bp.val[1];
        int8x16_t b2 = _bp.val[2];
        int8x16_t b3 = _bp.val[3];
        fe_i8mm_k8_acc_rp(A_rp_tile, g8, b0, b1, b2, b3,
                          &c00, &c01, &c02, &c03,
                          &c10, &c11, &c12, &c13,
                          &c20, &c21, &c22, &c23,
                          &c30, &c31, &c32, &c33);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_i8mm_fused_write_rowpair_track(0, c00, c01, c02, c03, vs_lo, vs_hi,
                                       bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_i8mm_fused_write_rowpair_track(1, c10, c11, c12, c13, vs_lo, vs_hi,
                                       bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_i8mm_fused_write_rowpair_track(2, c20, c21, c22, c23, vs_lo, vs_hi,
                                       bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_i8mm_fused_write_rowpair_track(3, c30, c31, c32, c33, vs_lo, vs_hi,
                                       bias_n0, C, ldc, vmax_lo, vmax_hi);
}

__attribute__((hot))
void qgemm_i8mm_fp32_fused_track_maxabs(int M, int N, int K,
                                          const int8_t *A_q, const int8_t *Bp,
                                          const float *combined_scale,
                                          const float *bias,
                                          float *C, int ldc,
                                          int32_t *c32_tail,
                                          float *max_abs_out) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    if ((K & 7) != 0) {
        /* DOTPROD fp32-fused has no max-abs variant: run plain, scan after. */
        qgemm_dotprod_fp32_fused(M, N, K, A_q, Bp, combined_scale, bias,
                              C, ldc, 0, c32_tail);
        float mx = 0.0f;
        if (ldc == N) {
            mx = fe_qg_max_abs(C, M * N);
        } else {
            for (int r = 0; r < M; ++r) {
                float rm = fe_qg_max_abs(C + (size_t)r * ldc, N);
                if (rm > mx) mx = rm;
            }
        }
        *max_abs_out = mx;
        return;
    }
    float32x4_t vmax_lo = vdupq_n_f32(0.0f);
    float32x4_t vmax_hi = vdupq_n_f32(0.0f);
    int k8_groups = K / 8;
    fe_i8mm_repack_a_rp(A_q, M, K, K, fe_i8mm_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_i8mm_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_i8mm_fused_track_rp(
                k8_groups, A_rp_tile,
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
 * qgemm_i8mm_gru_full_fused: fuse W_ih, W_hh and gate into one pass.
 *
 * Per N-block:
 *   1. I8MM W_ih @ x_tile -> dequant to ih_tile (fp32, L1d-resident)
 *   2. I8MM W_hh @ h_tile -> combine with ih_tile + bsum -> sigmoid/tanh
 * Three loops (r, z, n) reuse the same body; n-gate also consumes r_band
 * and z_band.
 */
#if defined(__aarch64__)

/* I8MM 8x8 tile body for the GRU full-fusion kernel (row-pair A). */
#define I8MM_8X8_TILE_LOCAL_RP(k8g, A_rp_tile, Bp,                            \
                                c00,c01,c02,c03, c10,c11,c12,c13,              \
                                c20,c21,c22,c23, c30,c31,c32_,c33)             \
    do {                                                                       \
        c00 = c01 = c02 = c03 = vdupq_n_s32(0);                                \
        c10 = c11 = c12 = c13 = vdupq_n_s32(0);                                \
        c20 = c21 = c22 = c23 = vdupq_n_s32(0);                                \
        c30 = c31 = c32_ = c33 = vdupq_n_s32(0);                               \
        for (int g8 = 0; g8 < (k8g); ++g8) {                                   \
            int8x16x4_t _bp = vld1q_s8_x4((Bp) + g8 * 64);                     \
            int8x16_t _b0 = _bp.val[0];                                        \
            int8x16_t _b1 = _bp.val[1];                                        \
            int8x16_t _b2 = _bp.val[2];                                        \
            int8x16_t _b3 = _bp.val[3];                                        \
            fe_i8mm_k8_acc_rp((A_rp_tile), g8, _b0, _b1, _b2, _b3,            \
                &c00,&c01,&c02,&c03, &c10,&c11,&c12,&c13,                      \
                &c20,&c21,&c22,&c23, &c30,&c31,&c32_,&c33);                    \
        }                                                                       \
    } while (0)

/* dequant one row-pair of int32 I8MM acc to fp32 ih_band lo/hi */
static inline void fe_i8mm_dq_rowpair_to_ih(
        int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3,
        float32x4_t vs_lo, float32x4_t vs_hi,
        float32x4_t vb_lo, float32x4_t vb_hi,
        float *ih_tile_rp /* [4 vecs: r0_lo, r0_hi, r1_lo, r1_hi]*/) {
    int32x4_t i0_lo, i0_hi, i1_lo, i1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &i0_lo, &i0_hi, &i1_lo, &i1_hi);
    float32x4_t r0_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i0_lo), vs_lo);
    float32x4_t r0_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i0_hi), vs_hi);
    float32x4_t r1_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i1_lo), vs_lo);
    float32x4_t r1_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i1_hi), vs_hi);
    vst1q_f32(ih_tile_rp + 0,  r0_lo);
    vst1q_f32(ih_tile_rp + 4,  r0_hi);
    vst1q_f32(ih_tile_rp + 8,  r1_lo);
    vst1q_f32(ih_tile_rp + 12, r1_hi);
}

/* r/z gate combine: add(add(ih, hh), bsum) -> sigmoid. */
static inline void fe_i8mm_rzgate_rowpair_from_tile(
        int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3,
        float32x4_t vs_lo, float32x4_t vs_hi,
        float32x4_t vb_lo, float32x4_t vb_hi,
        const float *ih_tile_rp,
        const float *bsum_n,
        float *band, int ld_band,
        int rp, int n_off) {
    int32x4_t i0_lo, i0_hi, i1_lo, i1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &i0_lo, &i0_hi, &i1_lo, &i1_hi);
    float32x4_t hh0_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i0_lo), vs_lo);
    float32x4_t hh0_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i0_hi), vs_hi);
    float32x4_t hh1_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i1_lo), vs_lo);
    float32x4_t hh1_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i1_hi), vs_hi);
    float32x4_t ih0_lo = vld1q_f32(ih_tile_rp + 0);
    float32x4_t ih0_hi = vld1q_f32(ih_tile_rp + 4);
    float32x4_t ih1_lo = vld1q_f32(ih_tile_rp + 8);
    float32x4_t ih1_hi = vld1q_f32(ih_tile_rp + 12);
    float32x4_t bsum_lo = vld1q_f32(bsum_n + 0);
    float32x4_t bsum_hi = vld1q_f32(bsum_n + 4);
    float32x4_t p0_lo = vaddq_f32(vaddq_f32(ih0_lo, hh0_lo), bsum_lo);
    float32x4_t p0_hi = vaddq_f32(vaddq_f32(ih0_hi, hh0_hi), bsum_hi);
    float32x4_t p1_lo = vaddq_f32(vaddq_f32(ih1_lo, hh1_lo), bsum_lo);
    float32x4_t p1_hi = vaddq_f32(vaddq_f32(ih1_hi, hh1_hi), bsum_hi);
    float32x4_t g0_lo = FE_SIGMOIDF4(p0_lo);
    float32x4_t g0_hi = FE_SIGMOIDF4(p0_hi);
    float32x4_t g1_lo = FE_SIGMOIDF4(p1_lo);
    float32x4_t g1_hi = FE_SIGMOIDF4(p1_hi);
    vst1q_f32(band + (2 * rp + 0) * ld_band + n_off + 0, g0_lo);
    vst1q_f32(band + (2 * rp + 0) * ld_band + n_off + 4, g0_hi);
    vst1q_f32(band + (2 * rp + 1) * ld_band + n_off + 0, g1_lo);
    vst1q_f32(band + (2 * rp + 1) * ld_band + n_off + 4, g1_hi);
}

/* I8MM fp16 inout — rowpair variant: gru_h stored fp16, ngate epilogue
 * dual-stores (fp32 scratch for rnn_fc + fp16 storage update). */
static inline void fe_i8mm_ngate_rowpair_from_tile_fp16inout(
        int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3,
        float32x4_t vs_lo, float32x4_t vs_hi,
        float32x4_t vb_lo, float32x4_t vb_hi,
        const float *ih_tile_rp,
        const float *bn_i_n, const float *bn_h_n,
        const float *r_band, const float *z_band,
        int ld_band,
        uint16_t *h_row0_fp16, uint16_t *h_row1_fp16,
        float *h_out_row0, float *h_out_row1,
        int rp, int n_off) {
    int32x4_t i0_lo, i0_hi, i1_lo, i1_hi;
    fe_i8mm_gather_rowpair(c0, c1, c2, c3, &i0_lo, &i0_hi, &i1_lo, &i1_hi);
    float32x4_t hh0_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i0_lo), vs_lo);
    float32x4_t hh0_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i0_hi), vs_hi);
    float32x4_t hh1_lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(i1_lo), vs_lo);
    float32x4_t hh1_hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(i1_hi), vs_hi);
    float32x4_t ih0_lo = vld1q_f32(ih_tile_rp + 0);
    float32x4_t ih0_hi = vld1q_f32(ih_tile_rp + 4);
    float32x4_t ih1_lo = vld1q_f32(ih_tile_rp + 8);
    float32x4_t ih1_hi = vld1q_f32(ih_tile_rp + 12);
    float32x4_t bi_lo = vld1q_f32(bn_i_n + 0);
    float32x4_t bi_hi = vld1q_f32(bn_i_n + 4);
    float32x4_t bh_lo = vld1q_f32(bn_h_n + 0);
    float32x4_t bh_hi = vld1q_f32(bn_h_n + 4);
    float32x4_t ihp0_lo = vaddq_f32(ih0_lo, bi_lo);
    float32x4_t ihp0_hi = vaddq_f32(ih0_hi, bi_hi);
    float32x4_t ihp1_lo = vaddq_f32(ih1_lo, bi_lo);
    float32x4_t ihp1_hi = vaddq_f32(ih1_hi, bi_hi);
    float32x4_t hhp0_lo = vaddq_f32(hh0_lo, bh_lo);
    float32x4_t hhp0_hi = vaddq_f32(hh0_hi, bh_hi);
    float32x4_t hhp1_lo = vaddq_f32(hh1_lo, bh_lo);
    float32x4_t hhp1_hi = vaddq_f32(hh1_hi, bh_hi);
    float32x4_t rb0_lo = vld1q_f32(r_band + (2 * rp + 0) * ld_band + n_off + 0);
    float32x4_t rb0_hi = vld1q_f32(r_band + (2 * rp + 0) * ld_band + n_off + 4);
    float32x4_t rb1_lo = vld1q_f32(r_band + (2 * rp + 1) * ld_band + n_off + 0);
    float32x4_t rb1_hi = vld1q_f32(r_band + (2 * rp + 1) * ld_band + n_off + 4);
    float32x4_t np0_lo = vfmaq_f32(ihp0_lo, rb0_lo, hhp0_lo);
    float32x4_t np0_hi = vfmaq_f32(ihp0_hi, rb0_hi, hhp0_hi);
    float32x4_t np1_lo = vfmaq_f32(ihp1_lo, rb1_lo, hhp1_lo);
    float32x4_t np1_hi = vfmaq_f32(ihp1_hi, rb1_hi, hhp1_hi);
    float32x4_t n0_lo  = FE_TANHF4(np0_lo);
    float32x4_t n0_hi  = FE_TANHF4(np0_hi);
    float32x4_t n1_lo  = FE_TANHF4(np1_lo);
    float32x4_t n1_hi  = FE_TANHF4(np1_hi);
    float32x4_t zb0_lo = vld1q_f32(z_band + (2 * rp + 0) * ld_band + n_off + 0);
    float32x4_t zb0_hi = vld1q_f32(z_band + (2 * rp + 0) * ld_band + n_off + 4);
    float32x4_t zb1_lo = vld1q_f32(z_band + (2 * rp + 1) * ld_band + n_off + 0);
    float32x4_t zb1_hi = vld1q_f32(z_band + (2 * rp + 1) * ld_band + n_off + 4);
    float32x4_t ho0_lo = fe_fp16_load4(h_row0_fp16 + n_off + 0);
    float32x4_t ho0_hi = fe_fp16_load4(h_row0_fp16 + n_off + 4);
    float32x4_t ho1_lo = fe_fp16_load4(h_row1_fp16 + n_off + 0);
    float32x4_t ho1_hi = fe_fp16_load4(h_row1_fp16 + n_off + 4);
    float32x4_t hn0_lo = vfmaq_f32(n0_lo, zb0_lo, vsubq_f32(ho0_lo, n0_lo));
    float32x4_t hn0_hi = vfmaq_f32(n0_hi, zb0_hi, vsubq_f32(ho0_hi, n0_hi));
    float32x4_t hn1_lo = vfmaq_f32(n1_lo, zb1_lo, vsubq_f32(ho1_lo, n1_lo));
    float32x4_t hn1_hi = vfmaq_f32(n1_hi, zb1_hi, vsubq_f32(ho1_hi, n1_hi));
    vst1q_f32(h_out_row0 + n_off + 0, hn0_lo);
    vst1q_f32(h_out_row0 + n_off + 4, hn0_hi);
    vst1q_f32(h_out_row1 + n_off + 0, hn1_lo);
    vst1q_f32(h_out_row1 + n_off + 4, hn1_hi);
    fe_fp16_store4(h_row0_fp16 + n_off + 0, hn0_lo);
    fe_fp16_store4(h_row0_fp16 + n_off + 4, hn0_hi);
    fe_fp16_store4(h_row1_fp16 + n_off + 0, hn1_lo);
    fe_fp16_store4(h_row1_fp16 + n_off + 4, hn1_hi);
}

__attribute__((hot))
void qgemm_i8mm_gru_full_fused_fp16inout(int M, int D,
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

    int32x4_t c00, c01, c02, c03, c10, c11, c12, c13;
    int32x4_t c20, c21, c22, c23, c30, c31, c32_, c33;

    /* row-pair pre-pack X and H once for the entire GRU call. */
    const int k8_groups = K / 8;
    fe_i8mm_repack_a_rp(Xq, M, K, ld_x, fe_i8mm_gru_x_rp_buf);
    fe_i8mm_repack_a_rp(Hq, M, K, ld_h, fe_i8mm_gru_h_rp_buf);

    for (int mr = 0; mr + MR <= M; mr += MR) {
        const int8_t *X_rp_tile        = fe_i8mm_gru_x_rp_buf + (size_t)mr * K;
        const int8_t *H_rp_tile        = fe_i8mm_gru_h_rp_buf + (size_t)mr * K;
        uint16_t     *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float        *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;

        /* r gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)nb * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)nb * weight_n_block_stride;
            int n_off = nb * NR;
            I8MM_8X8_TILE_LOCAL_RP(k8_groups, X_rp_tile, Bp_ih,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + n_off + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + n_off + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + n_off + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + n_off + 4);
            fe_i8mm_dq_rowpair_to_ih(c00,c01,c02,c03, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*16);
            fe_i8mm_dq_rowpair_to_ih(c10,c11,c12,c13, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*16);
            fe_i8mm_dq_rowpair_to_ih(c20,c21,c22,c23, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*16);
            fe_i8mm_dq_rowpair_to_ih(c30,c31,c32_,c33, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*16);

            I8MM_8X8_TILE_LOCAL_RP(k8_groups, H_rp_tile, Bp_hh,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + n_off + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + n_off + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + n_off + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + n_off + 4);
            fe_i8mm_rzgate_rowpair_from_tile(c00,c01,c02,c03, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*16, br_sum + n_off, r_band, ld_band, 0, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c10,c11,c12,c13, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*16, br_sum + n_off, r_band, ld_band, 1, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c20,c21,c22,c23, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*16, br_sum + n_off, r_band, ld_band, 2, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c30,c31,c32_,c33, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*16, br_sum + n_off, r_band, ld_band, 3, n_off);
        }

        /* z gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = D + n_off;
            I8MM_8X8_TILE_LOCAL_RP(k8_groups, X_rp_tile, Bp_ih,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_i8mm_dq_rowpair_to_ih(c00,c01,c02,c03, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*16);
            fe_i8mm_dq_rowpair_to_ih(c10,c11,c12,c13, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*16);
            fe_i8mm_dq_rowpair_to_ih(c20,c21,c22,c23, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*16);
            fe_i8mm_dq_rowpair_to_ih(c30,c31,c32_,c33, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*16);

            I8MM_8X8_TILE_LOCAL_RP(k8_groups, H_rp_tile, Bp_hh,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_i8mm_rzgate_rowpair_from_tile(c00,c01,c02,c03, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*16, bz_sum + n_off, z_band, ld_band, 0, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c10,c11,c12,c13, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*16, bz_sum + n_off, z_band, ld_band, 1, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c20,c21,c22,c23, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*16, bz_sum + n_off, z_band, ld_band, 2, n_off);
            fe_i8mm_rzgate_rowpair_from_tile(c30,c31,c32_,c33, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*16, bz_sum + n_off, z_band, ld_band, 3, n_off);
        }

        /* n gate + h_new (fp16 dual store) */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = 2 * D + n_off;
            I8MM_8X8_TILE_LOCAL_RP(k8_groups, X_rp_tile, Bp_ih,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_i8mm_dq_rowpair_to_ih(c00,c01,c02,c03, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*16);
            fe_i8mm_dq_rowpair_to_ih(c10,c11,c12,c13, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*16);
            fe_i8mm_dq_rowpair_to_ih(c20,c21,c22,c23, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*16);
            fe_i8mm_dq_rowpair_to_ih(c30,c31,c32_,c33, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*16);

            I8MM_8X8_TILE_LOCAL_RP(k8_groups, H_rp_tile, Bp_hh,
                                    c00,c01,c02,c03, c10,c11,c12,c13,
                                    c20,c21,c22,c23, c30,c31,c32_,c33);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_i8mm_ngate_rowpair_from_tile_fp16inout(c00,c01,c02,c03, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*16, bn_i + n_off, bn_h + n_off, r_band, z_band, ld_band, h_block_storage + 0 * ld_h_inout, h_block_storage + 1 * ld_h_inout, h_block_scratch + 0 * ld_h_out, h_block_scratch + 1 * ld_h_out, 0, n_off);
            fe_i8mm_ngate_rowpair_from_tile_fp16inout(c10,c11,c12,c13, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*16, bn_i + n_off, bn_h + n_off, r_band, z_band, ld_band, h_block_storage + 2 * ld_h_inout, h_block_storage + 3 * ld_h_inout, h_block_scratch + 2 * ld_h_out, h_block_scratch + 3 * ld_h_out, 1, n_off);
            fe_i8mm_ngate_rowpair_from_tile_fp16inout(c20,c21,c22,c23, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*16, bn_i + n_off, bn_h + n_off, r_band, z_band, ld_band, h_block_storage + 4 * ld_h_inout, h_block_storage + 5 * ld_h_inout, h_block_scratch + 4 * ld_h_out, h_block_scratch + 5 * ld_h_out, 2, n_off);
            fe_i8mm_ngate_rowpair_from_tile_fp16inout(c30,c31,c32_,c33, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*16, bn_i + n_off, bn_h + n_off, r_band, z_band, ld_band, h_block_storage + 6 * ld_h_inout, h_block_storage + 7 * ld_h_inout, h_block_scratch + 6 * ld_h_out, h_block_scratch + 7 * ld_h_out, 3, n_off);
        }
    }
}

#undef I8MM_8X8_TILE_LOCAL_RP

#endif /* aarch64 (gru_full_fused) */

#endif /* aarch64 */

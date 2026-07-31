/*
 * arm/qgemm_dotprod.c - ARMv8.2+ FEAT_DotProd (DOTPROD) int8 GEMM tier.
 *
 * 8x8 microkernel with K multiple of 4. Fused epilogues match the
 * 2-pass dequant FMA order for bit-identical results. Dispatched only
 * on hardware with FEAT_DotProd.
 */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

#define FE_QPOST_NEON 1
#include "../arch_kernels.h"
#include "../../internal/fe_qgemm.h"
#include "../../internal/fe_simd.h"
#include "../../internal/fe_fp16.h"  /* fp16 inout: load/store helpers */
#include "../qgemm_simd_post.inl"
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* A row-quad pre-pack (32 B/k4-group): the DOTPROD analogue of the I8MM
 * tier's row-pair pre-pack.
 *
 * The legacy `fe_dotprod_row_acc` builds the SDOT `a` operand on the fly via
 * `vld1q_dup_s32` (LD1R) — one load+broadcast µop per row per k4-group, i.e.
 * 8 LD1R per k-group on an 8-row tile. On Apple M2 SDOT issues at ~4/cycle but
 * LD1R competes for the load ports, so the 8 broadcasts/k-group throttle the
 * kernel to ~62% of SDOT peak (measured 136 vs 220 GMAC/s).
 *
 * With A pre-packed so each k4-group holds all 8 rows' 4-byte quads
 * contiguously (32 B = [r0q|r1q|...|r7q]), the kernel loads two `vld1q_s8`
 * (rows 0-3 and 4-7) per k-group and feeds them to `vdotq_laneq_s32` lane
 * indices 0..3 — eliminating every LD1R. Integer products are identical
 * (a*b == b*a, same 4-element group order) so output is byte-for-byte
 * unchanged and cross-tier bit-identity holds. Current canonical M2
 * microbench is ~186 GMAC/s, roughly matching I8MM (~183 GMAC/s) and landing
 * around 85-87% of the measured SDOT/SMMLA register-resident ceiling.
 *
 * Per MR=8 tile the packed bytes = (K/4)*32 = 8*K B; total per call = M*K
 * (same as original A). Current Medium max M*K is 128*288; the static
 * scratch bound is 256*FE_QGEMM_MAX_K. GRU keeps dedicated X/H buffers
 * (two simultaneous inputs, each <= D*D). */
#define FE_DOTPROD_MAX_A_MK   (256 * FE_QGEMM_MAX_K)
#define FE_DOTPROD_GRU_MAX_MK (FE_QGEMM_MAX_GRU_D * FE_QGEMM_MAX_GRU_D)

static int8_t fe_dotprod_a_rp_buf[FE_DOTPROD_MAX_A_MK];
static int8_t fe_dotprod_gru_x_rp_buf[FE_DOTPROD_GRU_MAX_MK];
static int8_t fe_dotprod_gru_h_rp_buf[FE_DOTPROD_GRU_MAX_MK];

/* Pre-fault the row-quad BSS scratch pages at init so the first-touch
 * (COW-zero) page faults happen here, not in the steady-state kernel. */
void qgemm_arm_dotprod_prefault_buffers(void) {
    const size_t PAGE = 4096;
    volatile unsigned char *p;
    p = (volatile unsigned char *)fe_dotprod_a_rp_buf;
    for (size_t i = 0; i < sizeof(fe_dotprod_a_rp_buf); i += PAGE) p[i] = 0;
    p = (volatile unsigned char *)fe_dotprod_gru_x_rp_buf;
    for (size_t i = 0; i < sizeof(fe_dotprod_gru_x_rp_buf); i += PAGE) p[i] = 0;
    p = (volatile unsigned char *)fe_dotprod_gru_h_rp_buf;
    for (size_t i = 0; i < sizeof(fe_dotprod_gru_h_rp_buf); i += PAGE) p[i] = 0;
}

/* Repack A from row-major [M, K] (stride lda) into the row-quad layout:
 * per full MR=8 tile, k4-group g occupies 32 contiguous bytes
 * [r0 quad | r1 quad | ... | r7 quad]. Production GEMMs are MR-aligned;
 * a non-aligned M trips fe_qgemm_tail_unsupported(). */
static inline void fe_dotprod_repack_a_rp(const int8_t *A_q, int M, int K,
                                          int lda, int8_t *A_rp) {
    int n_full_tiles = M / FE_QGEMM_MR;
    int k4_groups = K / 4;
    for (int t = 0; t < n_full_tiles; ++t) {
        int row0 = t * FE_QGEMM_MR;
        const int8_t *p0 = A_q + (size_t)(row0 + 0) * lda;
        const int8_t *p1 = A_q + (size_t)(row0 + 1) * lda;
        const int8_t *p2 = A_q + (size_t)(row0 + 2) * lda;
        const int8_t *p3 = A_q + (size_t)(row0 + 3) * lda;
        const int8_t *p4 = A_q + (size_t)(row0 + 4) * lda;
        const int8_t *p5 = A_q + (size_t)(row0 + 5) * lda;
        const int8_t *p6 = A_q + (size_t)(row0 + 6) * lda;
        const int8_t *p7 = A_q + (size_t)(row0 + 7) * lda;
        int8_t *tile_out = A_rp + (size_t)t * FE_QGEMM_MR * K;
        for (int g = 0; g < k4_groups; ++g) {
            /* Gather each row's 4-byte K-quad straight into a vector lane
             * (LD1 {Vt.S}[i] — pure vector load, no scalar GPR round-trip),
             * then two 16-byte vector stores. a0 = rows 0-3, a1 = rows 4-7,
             * matching the vdotq_laneq lane order in fe_dotprod_k4_acc_rp. */
            const int o = g * 4;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            a0 = vld1q_lane_s32((const int32_t *)(p0 + o), a0, 0);
            a0 = vld1q_lane_s32((const int32_t *)(p1 + o), a0, 1);
            a0 = vld1q_lane_s32((const int32_t *)(p2 + o), a0, 2);
            a0 = vld1q_lane_s32((const int32_t *)(p3 + o), a0, 3);
            a1 = vld1q_lane_s32((const int32_t *)(p4 + o), a1, 0);
            a1 = vld1q_lane_s32((const int32_t *)(p5 + o), a1, 1);
            a1 = vld1q_lane_s32((const int32_t *)(p6 + o), a1, 2);
            a1 = vld1q_lane_s32((const int32_t *)(p7 + o), a1, 3);
            vst1q_s8(tile_out + (size_t)g * 32,      vreinterpretq_s8_s32(a0));
            vst1q_s8(tile_out + (size_t)g * 32 + 16, vreinterpretq_s8_s32(a1));
        }
    }
}

/* Row-quad-packed accumulator: one k4-group, all 8 rows via vdotq_laneq.
 * A_rp_tile is the packed tile base; `g*32` advances to the next k4-group. */
static inline void fe_dotprod_k4_acc_rp(const int8_t *A_rp_tile, int g,
        int8x16_t b_lo, int8x16_t b_hi,
        int32x4_t *c00, int32x4_t *c01, int32x4_t *c10, int32x4_t *c11,
        int32x4_t *c20, int32x4_t *c21, int32x4_t *c30, int32x4_t *c31,
        int32x4_t *c40, int32x4_t *c41, int32x4_t *c50, int32x4_t *c51,
        int32x4_t *c60, int32x4_t *c61, int32x4_t *c70, int32x4_t *c71) {
    int8x16_t a0 = vld1q_s8(A_rp_tile + (size_t)g * 32);       /* rows 0-3 quads */
    int8x16_t a1 = vld1q_s8(A_rp_tile + (size_t)g * 32 + 16);  /* rows 4-7 quads */
    *c00 = vdotq_laneq_s32(*c00, b_lo, a0, 0); *c01 = vdotq_laneq_s32(*c01, b_hi, a0, 0);
    *c10 = vdotq_laneq_s32(*c10, b_lo, a0, 1); *c11 = vdotq_laneq_s32(*c11, b_hi, a0, 1);
    *c20 = vdotq_laneq_s32(*c20, b_lo, a0, 2); *c21 = vdotq_laneq_s32(*c21, b_hi, a0, 2);
    *c30 = vdotq_laneq_s32(*c30, b_lo, a0, 3); *c31 = vdotq_laneq_s32(*c31, b_hi, a0, 3);
    *c40 = vdotq_laneq_s32(*c40, b_lo, a1, 0); *c41 = vdotq_laneq_s32(*c41, b_hi, a1, 0);
    *c50 = vdotq_laneq_s32(*c50, b_lo, a1, 1); *c51 = vdotq_laneq_s32(*c51, b_hi, a1, 1);
    *c60 = vdotq_laneq_s32(*c60, b_lo, a1, 2); *c61 = vdotq_laneq_s32(*c61, b_hi, a1, 2);
    *c70 = vdotq_laneq_s32(*c70, b_lo, a1, 3); *c71 = vdotq_laneq_s32(*c71, b_hi, a1, 3);
}

static inline void qgemm_kernel_8x8_dotprod(int K,
                                         const int8_t *A_rp_tile,
                                         const int8_t *Bp,
                                         int32_t *C32, int ldc32) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
    int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
    int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
    int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        int8x16x2_t _bp = vld1q_s8_x2(Bp + (size_t)g * 32);
        int8x16_t b_lo = _bp.val[0];
        int8x16_t b_hi = _bp.val[1];
        fe_dotprod_k4_acc_rp(A_rp_tile, g, b_lo, b_hi,
                             &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31,
                             &c40, &c41, &c50, &c51, &c60, &c61, &c70, &c71);
    }

    vst1q_s32(C32 + 0 * ldc32 + 0, c00); vst1q_s32(C32 + 0 * ldc32 + 4, c01);
    vst1q_s32(C32 + 1 * ldc32 + 0, c10); vst1q_s32(C32 + 1 * ldc32 + 4, c11);
    vst1q_s32(C32 + 2 * ldc32 + 0, c20); vst1q_s32(C32 + 2 * ldc32 + 4, c21);
    vst1q_s32(C32 + 3 * ldc32 + 0, c30); vst1q_s32(C32 + 3 * ldc32 + 4, c31);
    vst1q_s32(C32 + 4 * ldc32 + 0, c40); vst1q_s32(C32 + 4 * ldc32 + 4, c41);
    vst1q_s32(C32 + 5 * ldc32 + 0, c50); vst1q_s32(C32 + 5 * ldc32 + 4, c51);
    vst1q_s32(C32 + 6 * ldc32 + 0, c60); vst1q_s32(C32 + 6 * ldc32 + 4, c61);
    vst1q_s32(C32 + 7 * ldc32 + 0, c70); vst1q_s32(C32 + 7 * ldc32 + 4, c71);
}

void qgemm_dotprod_int32(int M, int N, int K,
                      const int8_t *A_q, const int8_t *Bp,
                      int32_t *C32, int ldc32) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    /* one-shot row-quad repack of A; reused across all (mr, nr) tiles. */
    fe_dotprod_repack_a_rp(A_q, M, K, K, fe_dotprod_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_dotprod_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_dotprod(K, A_rp_tile,
                                  Bp + (size_t)bn * k4_groups * NR * 4,
                                  C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* K-fixed specialisation for MHSA Q@K^T (K=20): the k4 loop fully unrolls. */
__attribute__((always_inline))
static inline void qgemm_kernel_8x8_dotprod_k20(const int8_t *A_rp_tile,
                                             const int8_t *Bp,
                                             int32_t *C32, int ldc32) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
    int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
    int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
    int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

    /* K = 20, k4_groups = 5, unrolled (row-quad packed A). */
    #define DOTPROD_K20_GROUP(G) do {                                          \
        int8x16x2_t _bp = vld1q_s8_x2(Bp + (size_t)(G) * 32);                  \
        int8x16_t b_lo = _bp.val[0];                                           \
        int8x16_t b_hi = _bp.val[1];                                           \
        fe_dotprod_k4_acc_rp(A_rp_tile, (G), b_lo, b_hi,                       \
                             &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31,   \
                             &c40, &c41, &c50, &c51, &c60, &c61, &c70, &c71);  \
    } while (0)
    DOTPROD_K20_GROUP(0);
    DOTPROD_K20_GROUP(1);
    DOTPROD_K20_GROUP(2);
    DOTPROD_K20_GROUP(3);
    DOTPROD_K20_GROUP(4);
    #undef DOTPROD_K20_GROUP

    vst1q_s32(C32 + 0 * ldc32 + 0, c00); vst1q_s32(C32 + 0 * ldc32 + 4, c01);
    vst1q_s32(C32 + 1 * ldc32 + 0, c10); vst1q_s32(C32 + 1 * ldc32 + 4, c11);
    vst1q_s32(C32 + 2 * ldc32 + 0, c20); vst1q_s32(C32 + 2 * ldc32 + 4, c21);
    vst1q_s32(C32 + 3 * ldc32 + 0, c30); vst1q_s32(C32 + 3 * ldc32 + 4, c31);
    vst1q_s32(C32 + 4 * ldc32 + 0, c40); vst1q_s32(C32 + 4 * ldc32 + 4, c41);
    vst1q_s32(C32 + 5 * ldc32 + 0, c50); vst1q_s32(C32 + 5 * ldc32 + 4, c51);
    vst1q_s32(C32 + 6 * ldc32 + 0, c60); vst1q_s32(C32 + 6 * ldc32 + 4, c61);
    vst1q_s32(C32 + 7 * ldc32 + 0, c70); vst1q_s32(C32 + 7 * ldc32 + 4, c71);
}

void qgemm_dotprod_int32_k20(int M, int N,
                          const int8_t *A_q, const int8_t *Bp,
                          int32_t *C32, int ldc32) {
    enum { K = 20, MR = FE_QGEMM_MR, NR = FE_QGEMM_NR };
    /* k4_groups for the weight pack stride. */
    enum { k4_groups = (K + 3) / 4 };
    /* row-quad repack of A (K=20 -> 5 k4-groups per tile). */
    fe_dotprod_repack_a_rp(A_q, M, K, K, fe_dotprod_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_dotprod_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_dotprod_k20(A_rp_tile,
                                       Bp + (size_t)bn * k4_groups * NR * 4,
                                       C32 + (size_t)mr * ldc32 + nr, ldc32);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/* fp32-out fused: DOTPROD + dequant + bias + optional SiLU. */
static inline void fe_dotprod_fused_store_row(int R,
                                           int32x4_t acc_lo, int32x4_t acc_hi,
                                           float32x4_t vs_lo, float32x4_t vs_hi,
                                           const float *bias_n0,
                                           int act_silu,
                                           float *C, int ldc) {
    float32x4_t vl = vcvtq_f32_s32(acc_lo);
    float32x4_t vh = vcvtq_f32_s32(acc_hi);
    if (bias_n0) {
        vl = vfmaq_f32(vld1q_f32(bias_n0 + 0), vl, vs_lo);
        vh = vfmaq_f32(vld1q_f32(bias_n0 + 4), vh, vs_hi);
    } else {
        vl = vmulq_f32(vl, vs_lo);
        vh = vmulq_f32(vh, vs_hi);
    }
    if (act_silu) {
        vl = fe_mul(vl, FE_SIGMOIDF4(vl));
        vh = fe_mul(vh, FE_SIGMOIDF4(vh));
    }
    vst1q_f32(C + R * ldc + 0, vl);
    vst1q_f32(C + R * ldc + 4, vh);
}

static inline void qgemm_kernel_8x8_dotprod_fused(int K,
                                               const int8_t *A_rp_tile,
                                               const int8_t *Bp,
                                               const float *combined_scale_n0,
                                               const float *bias_n0,
                                               float *C, int ldc,
                                               int act_silu) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
    int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
    int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
    int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        int8x16x2_t _bp = vld1q_s8_x2(Bp + (size_t)g * 32);
        int8x16_t b_lo = _bp.val[0];
        int8x16_t b_hi = _bp.val[1];
        fe_dotprod_k4_acc_rp(A_rp_tile, g, b_lo, b_hi,
                             &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31,
                             &c40, &c41, &c50, &c51, &c60, &c61, &c70, &c71);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_dotprod_fused_store_row(0, c00, c01, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(1, c10, c11, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(2, c20, c21, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(3, c30, c31, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(4, c40, c41, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(5, c50, c51, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(6, c60, c61, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
    fe_dotprod_fused_store_row(7, c70, c71, vs_lo, vs_hi, bias_n0, act_silu, C, ldc);
}


void qgemm_dotprod_fp32_fused(int M, int N, int K,
                           const int8_t *A_q, const int8_t *Bp,
                           const float *combined_scale, const float *bias,
                           float *C, int ldc,
                           int act_silu, int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    fe_dotprod_repack_a_rp(A_q, M, K, K, fe_dotprod_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_dotprod_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_dotprod_fused(
                K, A_rp_tile,
                Bp + (size_t)bn * k4_groups * NR * 4,
                combined_scale + nr,
                bias ? bias + nr : NULL,
                C + (size_t)mr * ldc + nr, ldc, act_silu);
        }
        if (nr < N) fe_qgemm_tail_unsupported();
    }
    if (mr < M) fe_qgemm_tail_unsupported();
}

/*
 * fp32-out fused, accumulating epilogue.
 *   C[m,n] += bias[n] + scale[n] * c32[m,n]
 * Order: tmp = bias + scale*c32 (vfmaq), then C += tmp (vaddq).
 */
static inline void fe_dotprod_fused_store_row_acc(int R,
                                                int32x4_t acc_lo, int32x4_t acc_hi,
                                                float32x4_t vs_lo, float32x4_t vs_hi,
                                                const float *bias_n0,
                                                float *C, int ldc) {
    float32x4_t vl = vcvtq_f32_s32(acc_lo);
    float32x4_t vh = vcvtq_f32_s32(acc_hi);
    if (bias_n0) {
        vl = vfmaq_f32(vld1q_f32(bias_n0 + 0), vl, vs_lo);
        vh = vfmaq_f32(vld1q_f32(bias_n0 + 4), vh, vs_hi);
    } else {
        vl = vmulq_f32(vl, vs_lo);
        vh = vmulq_f32(vh, vs_hi);
    }
    float32x4_t cl = vld1q_f32(C + R * ldc + 0);
    float32x4_t ch = vld1q_f32(C + R * ldc + 4);
    vst1q_f32(C + R * ldc + 0, vaddq_f32(cl, vl));
    vst1q_f32(C + R * ldc + 4, vaddq_f32(ch, vh));
}

static inline void qgemm_kernel_8x8_dotprod_fused_acc(int K,
                                                    const int8_t *A_rp_tile,
                                                    const int8_t *Bp,
                                                    const float *combined_scale_n0,
                                                    const float *bias_n0,
                                                    float *C, int ldc) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
    int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
    int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
    int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        int8x16x2_t _bp = vld1q_s8_x2(Bp + (size_t)g * 32);
        int8x16_t b_lo = _bp.val[0];
        int8x16_t b_hi = _bp.val[1];
        fe_dotprod_k4_acc_rp(A_rp_tile, g, b_lo, b_hi,
                             &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31,
                             &c40, &c41, &c50, &c51, &c60, &c61, &c70, &c71);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_dotprod_fused_store_row_acc(0, c00, c01, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(1, c10, c11, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(2, c20, c21, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(3, c30, c31, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(4, c40, c41, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(5, c50, c51, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(6, c60, c61, vs_lo, vs_hi, bias_n0, C, ldc);
    fe_dotprod_fused_store_row_acc(7, c70, c71, vs_lo, vs_hi, bias_n0, C, ldc);
}


__attribute__((hot))
void qgemm_dotprod_fp32_fused_acc(int M, int N, int K,
                                const int8_t *A_q, const int8_t *Bp,
                                const float *combined_scale, const float *bias,
                                float *C, int ldc,
                                int32_t *c32_tail) {
    const int MR = FE_QGEMM_MR;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    fe_dotprod_repack_a_rp(A_q, M, K, K, fe_dotprod_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_dotprod_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_dotprod_fused_acc(K, A_rp_tile,
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
 * fp32-out fused with max-abs tracking. Folds max|C[m,n]| into the
 * dequant epilogue. No SiLU; caller is MHSA QKV projection.
 */
static inline void fe_dotprod_fused_store_row_track(int R,
                                                  int32x4_t acc_lo, int32x4_t acc_hi,
                                                  float32x4_t vs_lo, float32x4_t vs_hi,
                                                  const float *bias_n0,
                                                  float *C, int ldc,
                                                  float32x4_t *vmax_lo,
                                                  float32x4_t *vmax_hi) {
    float32x4_t vl = vcvtq_f32_s32(acc_lo);
    float32x4_t vh = vcvtq_f32_s32(acc_hi);
    if (bias_n0) {
        vl = vfmaq_f32(vld1q_f32(bias_n0 + 0), vl, vs_lo);
        vh = vfmaq_f32(vld1q_f32(bias_n0 + 4), vh, vs_hi);
    } else {
        vl = vmulq_f32(vl, vs_lo);
        vh = vmulq_f32(vh, vs_hi);
    }
    vst1q_f32(C + R * ldc + 0, vl);
    vst1q_f32(C + R * ldc + 4, vh);
    *vmax_lo = vmaxq_f32(*vmax_lo, vabsq_f32(vl));
    *vmax_hi = vmaxq_f32(*vmax_hi, vabsq_f32(vh));
}

static inline void qgemm_kernel_8x8_dotprod_fused_track(int K,
                                                      const int8_t *A_rp_tile,
                                                      const int8_t *Bp,
                                                      const float *combined_scale_n0,
                                                      const float *bias_n0,
                                                      float *C, int ldc,
                                                      float32x4_t *vmax_lo,
                                                      float32x4_t *vmax_hi) {
    int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
    int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
    int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
    int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
    int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
    int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
    int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
    int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

    int k4_groups = K / 4;
    for (int g = 0; g < k4_groups; ++g) {
        int8x16x2_t _bp = vld1q_s8_x2(Bp + (size_t)g * 32);
        int8x16_t b_lo = _bp.val[0];
        int8x16_t b_hi = _bp.val[1];
        fe_dotprod_k4_acc_rp(A_rp_tile, g, b_lo, b_hi,
                             &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31,
                             &c40, &c41, &c50, &c51, &c60, &c61, &c70, &c71);
    }

    float32x4_t vs_lo = vld1q_f32(combined_scale_n0 + 0);
    float32x4_t vs_hi = vld1q_f32(combined_scale_n0 + 4);
    fe_dotprod_fused_store_row_track(0, c00, c01, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(1, c10, c11, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(2, c20, c21, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(3, c30, c31, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(4, c40, c41, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(5, c50, c51, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(6, c60, c61, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
    fe_dotprod_fused_store_row_track(7, c70, c71, vs_lo, vs_hi, bias_n0, C, ldc, vmax_lo, vmax_hi);
}

__attribute__((hot))
void qgemm_dotprod_fp32_fused_track_maxabs(int M, int N, int K,
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
    fe_dotprod_repack_a_rp(A_q, M, K, K, fe_dotprod_a_rp_buf);
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        const int8_t *A_rp_tile = fe_dotprod_a_rp_buf + (size_t)mr * K;
        int nr = 0, bn = 0;
        for (; nr + NR <= N; nr += NR, ++bn) {
            qgemm_kernel_8x8_dotprod_fused_track(K, A_rp_tile,
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
 * qgemm_dotprod_gru_full_fused: fuse W_ih, W_hh and gate into one pass.
 * DOTPROD pass for W_ih dequants to a per-tile fp32 ih scratch; DOTPROD pass
 * for W_hh combines with ih + bsum register-resident.
 */

/* tile body over row-quad pre-packed A (A_rp_tile = 8*K packed bytes). */
#define DOTPROD_8X8_TILE_BODY(K, A_rp_tile, Bp,                              \
                           c00,c01,c10,c11,c20,c21,c30,c31,                 \
                           c40,c41,c50,c51,c60,c61,c70,c71)                 \
    do {                                                                     \
        c00 = c01 = c10 = c11 = vdupq_n_s32(0);                              \
        c20 = c21 = c30 = c31 = vdupq_n_s32(0);                              \
        c40 = c41 = c50 = c51 = vdupq_n_s32(0);                              \
        c60 = c61 = c70 = c71 = vdupq_n_s32(0);                              \
        int _k4g = (K) / 4;                                                  \
        for (int _g = 0; _g < _k4g; ++_g) {                                  \
            int8x16x2_t _bp = vld1q_s8_x2((Bp) + (size_t)_g * 32);           \
            fe_dotprod_k4_acc_rp((A_rp_tile), _g, _bp.val[0], _bp.val[1],    \
                &c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,                     \
                &c40,&c41,&c50,&c51,&c60,&c61,&c70,&c71);                    \
        }                                                                    \
    } while (0)

/* dequant one DOTPROD row (8 lanes = lo + hi) to fp32 ih_tile_row. */
static inline void fe_dotprod_dq_row_to_ih(int32x4_t acc_lo, int32x4_t acc_hi,
                                         float32x4_t vs_lo, float32x4_t vs_hi,
                                         float32x4_t vb_lo, float32x4_t vb_hi,
                                         float *ih_tile_row) {
    float32x4_t lo = vfmaq_f32(vb_lo, vcvtq_f32_s32(acc_lo), vs_lo);
    float32x4_t hi = vfmaq_f32(vb_hi, vcvtq_f32_s32(acc_hi), vs_hi);
    vst1q_f32(ih_tile_row + 0, lo);
    vst1q_f32(ih_tile_row + 4, hi);
}

/* r/z gate combine: add(add(ih, hh), bsum) -> sigmoid. */
static inline void fe_dotprod_rzgate_from_tile(int32x4_t acc_lo, int32x4_t acc_hi,
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

/* DOTPROD fp16 inout variant: gru_h stored fp16, ngate epilogue dual-stores
 * (fp32 scratch for rnn_fc + fp16 storage update). */
static inline void fe_dotprod_ngate_from_tile_fp16inout(
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
void qgemm_dotprod_gru_full_fused_fp16inout(int M, int D,
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
        const int8_t *X_block          = Xq + (size_t)mr * ld_x;
        const int8_t *H_block          = Hq + (size_t)mr * ld_h;
        uint16_t     *h_block_storage  = h_inout_fp16 + (size_t)mr * ld_h_inout;
        float        *h_block_scratch  = h_out_scratch + (size_t)mr * ld_h_out;
        /* pre-pack X and H tiles once; reused across all 3 gates. */
        fe_dotprod_repack_a_rp(X_block, MR, K, ld_x, fe_dotprod_gru_x_rp_buf);
        fe_dotprod_repack_a_rp(H_block, MR, K, ld_h, fe_dotprod_gru_h_rp_buf);
        const int8_t *x_rp = fe_dotprod_gru_x_rp_buf;
        const int8_t *h_rp = fe_dotprod_gru_h_rp_buf;

        /* r gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)nb * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)nb * weight_n_block_stride;
            int n_off = nb * NR;
            DOTPROD_8X8_TILE_BODY(K, x_rp, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + n_off + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + n_off + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + n_off + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + n_off + 4);
            fe_dotprod_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_dotprod_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_dotprod_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_dotprod_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_dotprod_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_dotprod_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_dotprod_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_dotprod_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            DOTPROD_8X8_TILE_BODY(K, h_rp, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + n_off + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + n_off + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + n_off + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + n_off + 4);
            fe_dotprod_rzgate_from_tile(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, br_sum + n_off, r_band, n_off, ld_band, 0);
            fe_dotprod_rzgate_from_tile(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, br_sum + n_off, r_band, n_off, ld_band, 1);
            fe_dotprod_rzgate_from_tile(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, br_sum + n_off, r_band, n_off, ld_band, 2);
            fe_dotprod_rzgate_from_tile(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, br_sum + n_off, r_band, n_off, ld_band, 3);
            fe_dotprod_rzgate_from_tile(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, br_sum + n_off, r_band, n_off, ld_band, 4);
            fe_dotprod_rzgate_from_tile(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, br_sum + n_off, r_band, n_off, ld_band, 5);
            fe_dotprod_rzgate_from_tile(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, br_sum + n_off, r_band, n_off, ld_band, 6);
            fe_dotprod_rzgate_from_tile(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, br_sum + n_off, r_band, n_off, ld_band, 7);
        }

        /* z gate */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = D + n_off;
            DOTPROD_8X8_TILE_BODY(K, x_rp, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_dotprod_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_dotprod_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_dotprod_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_dotprod_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_dotprod_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_dotprod_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_dotprod_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_dotprod_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            DOTPROD_8X8_TILE_BODY(K, h_rp, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_dotprod_rzgate_from_tile(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, bz_sum + n_off, z_band, n_off, ld_band, 0);
            fe_dotprod_rzgate_from_tile(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, bz_sum + n_off, z_band, n_off, ld_band, 1);
            fe_dotprod_rzgate_from_tile(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, bz_sum + n_off, z_band, n_off, ld_band, 2);
            fe_dotprod_rzgate_from_tile(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, bz_sum + n_off, z_band, n_off, ld_band, 3);
            fe_dotprod_rzgate_from_tile(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, bz_sum + n_off, z_band, n_off, ld_band, 4);
            fe_dotprod_rzgate_from_tile(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, bz_sum + n_off, z_band, n_off, ld_band, 5);
            fe_dotprod_rzgate_from_tile(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, bz_sum + n_off, z_band, n_off, ld_band, 6);
            fe_dotprod_rzgate_from_tile(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, bz_sum + n_off, z_band, n_off, ld_band, 7);
        }

        /* n gate + h_new (fp16 dual store) */
        for (int nb = 0; nb < N_blocks_per_gate; ++nb) {
            const int8_t *Bp_ih = Wq_ih + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            const int8_t *Bp_hh = Wq_hh + (size_t)(2 * N_blocks_per_gate + nb) * weight_n_block_stride;
            int n_off = nb * NR;
            int N_global = 2 * D + n_off;
            DOTPROD_8X8_TILE_BODY(K, x_rp, Bp_ih,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_ih_lo = vld1q_f32(combined_ih + N_global + 0);
            float32x4_t vs_ih_hi = vld1q_f32(combined_ih + N_global + 4);
            float32x4_t vb_ih_lo = vld1q_f32(bias_eff_ih + N_global + 0);
            float32x4_t vb_ih_hi = vld1q_f32(bias_eff_ih + N_global + 4);
            fe_dotprod_dq_row_to_ih(c00,c01, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 0*NR);
            fe_dotprod_dq_row_to_ih(c10,c11, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 1*NR);
            fe_dotprod_dq_row_to_ih(c20,c21, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 2*NR);
            fe_dotprod_dq_row_to_ih(c30,c31, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 3*NR);
            fe_dotprod_dq_row_to_ih(c40,c41, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 4*NR);
            fe_dotprod_dq_row_to_ih(c50,c51, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 5*NR);
            fe_dotprod_dq_row_to_ih(c60,c61, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 6*NR);
            fe_dotprod_dq_row_to_ih(c70,c71, vs_ih_lo,vs_ih_hi, vb_ih_lo,vb_ih_hi, ih_tile + 7*NR);

            DOTPROD_8X8_TILE_BODY(K, h_rp, Bp_hh,
                               c00,c01,c10,c11,c20,c21,c30,c31,
                               c40,c41,c50,c51,c60,c61,c70,c71);
            float32x4_t vs_hh_lo = vld1q_f32(combined_hh + N_global + 0);
            float32x4_t vs_hh_hi = vld1q_f32(combined_hh + N_global + 4);
            float32x4_t vb_hh_lo = vld1q_f32(bias_eff_hh + N_global + 0);
            float32x4_t vb_hh_hi = vld1q_f32(bias_eff_hh + N_global + 4);
            fe_dotprod_ngate_from_tile_fp16inout(c00,c01, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 0*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 0 * ld_h_inout, h_block_scratch + 0 * ld_h_out, 0);
            fe_dotprod_ngate_from_tile_fp16inout(c10,c11, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 1*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 1 * ld_h_inout, h_block_scratch + 1 * ld_h_out, 1);
            fe_dotprod_ngate_from_tile_fp16inout(c20,c21, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 2*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 2 * ld_h_inout, h_block_scratch + 2 * ld_h_out, 2);
            fe_dotprod_ngate_from_tile_fp16inout(c30,c31, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 3*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 3 * ld_h_inout, h_block_scratch + 3 * ld_h_out, 3);
            fe_dotprod_ngate_from_tile_fp16inout(c40,c41, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 4*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 4 * ld_h_inout, h_block_scratch + 4 * ld_h_out, 4);
            fe_dotprod_ngate_from_tile_fp16inout(c50,c51, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 5*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 5 * ld_h_inout, h_block_scratch + 5 * ld_h_out, 5);
            fe_dotprod_ngate_from_tile_fp16inout(c60,c61, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 6*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 6 * ld_h_inout, h_block_scratch + 6 * ld_h_out, 6);
            fe_dotprod_ngate_from_tile_fp16inout(c70,c71, vs_hh_lo,vs_hh_hi, vb_hh_lo,vb_hh_hi, ih_tile + 7*NR, bn_i + n_off, bn_h + n_off, r_band, z_band, n_off, ld_band, h_block_storage + 7 * ld_h_inout, h_block_scratch + 7 * ld_h_out, 7);
        }
    }
}

#undef DOTPROD_8X8_TILE_BODY

#endif /* aarch64 */

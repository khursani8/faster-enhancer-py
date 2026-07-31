/*
 * Single-step GRU (PyTorch linear_before_reset=1, gate order [r, z, n]):
 *   r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *   z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *   n  = tanh   ((W_in x + b_in) + r * (W_hn h + b_hn))
 *   h' = n + z * (h - n)
 *
 * Every supported tier has a fused kernel for W_ih x, W_hh h and the gate
 * update; the old fp32 ih_batch path is gone.
 */
#include "fe_internal.h"
#include "fe_sgemm.h"
#include "fe_qgemm.h"
#include "fe_simd.h"
#include "fe_fp16.h"
#include "fe_profile.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_dispatch.h"
#include "qgemm/arch_kernels.h"
#include "qgemm/qgemm_simd_post.inl"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef FE_GRU_MAX_D3
#define FE_GRU_MAX_D3 (3 * FE_C2)
#endif

/* Per-call dequant scratch shared across tiers. Engine is single-threaded. */
static float g_gru_combined_ih[FE_GRU_MAX_D3];
static float g_gru_bias_eff_ih[FE_GRU_MAX_D3];
static float g_gru_combined_hh[FE_GRU_MAX_D3];
static float g_gru_bias_eff_hh[FE_GRU_MAX_D3];

/* r/z gate scratch consumed by every full_fused kernel. Defined once here
 * (extern in arch_kernels.h) so the binary carries one copy. */
FE_ALIGN64 float fe_gru_r_band[FE_QGEMM_GRU_MR * FE_QGEMM_MAX_GRU_D];
FE_ALIGN64 float fe_gru_z_band[FE_QGEMM_GRU_MR * FE_QGEMM_MAX_GRU_D];


/*
 * fp16 inout GRU step. Every SIMD tier has a
 * dual-store ngate variant — kernel reads fp16 h_old via VCVTPH2PS /
 * FCVTL and writes the new state to BOTH fp16 storage (VCVTPS2PH /
 * FCVTN) and fp32 scratch (rnn_fc consumes the scratch).
 *
 *   h_inout_fp16  : recurrent state, fp16 storage (s->gru_h[blk]).
 *                   read for h_old, written with new state.
 *   h_out_scratch : fp32 scratch for rnn_fc; engine no longer packs.
 *
 * No scalar fallback: every supported target has a SIMD tier (aarch64 NEON;
 * x86 AVX2 tier with FMA3+F16C). Unsupported ISAs fail at fe_init (no
 * runtime slow path).
 */
/* Holds the per-call quantize + bias_eff setup that every tier needs
 * before dispatching its fp16-inout kernel. */
typedef struct {
    int8_t *x_q;
    int8_t *h_q;
    FeActQuant q_x;
    FeActQuant q_h;
    const float *br_sum;
    const float *bz_sum;
    const float *bn_i;
    const float *bn_h;
} FeGruFp16Setup;

static inline FeGruFp16Setup fe_gru_fp16_setup(FeGRU *g, const float *x,
                                                const uint16_t *h_inout_fp16,
                                                int freq, int8_t *aq) {
    FeGruFp16Setup s;
    const int D  = g->hidden_size;
    const int D3 = 3 * D;
    s.x_q = aq;
    s.h_q = aq + (size_t)freq * D;
    s.q_x = fe_quantize_activation     (x,             freq * D, s.x_q);
    s.q_h = fe_quantize_activation_fp16(h_inout_fp16,  freq * D, s.h_q);
    fe_qg_scale_vec(g->scales_ih, D3, s.q_x.scale, g_gru_combined_ih);
    fe_qg_scale_vec(g->scales_hh, D3, s.q_h.scale, g_gru_combined_hh);
    fe_qgemm_build_bias_eff_nobias(g_gru_combined_ih, g->row_sums_ih,
                                    (float)(128 - s.q_x.zp), D3,
                                    g_gru_bias_eff_ih);
    fe_qgemm_build_bias_eff_nobias(g_gru_combined_hh, g->row_sums_hh,
                                    (float)(128 - s.q_h.zp), D3,
                                    g_gru_bias_eff_hh);
    s.br_sum = g->b_combined;
    s.bz_sum = g->b_combined + D;
    s.bn_i   = g->b_ih + 2 * D;
    s.bn_h   = g->b_hh + 2 * D;
    return s;
}

static inline void fe_gru_fp16_track_act(FeGRU *g,
                                          FeActQuant q_x, FeActQuant q_h) {
    const float xmin = -(float)q_x.zp * q_x.scale;
    const float xmax = (255.0f - (float)q_x.zp) * q_x.scale;
    if (xmax > g->act_x.max_running) g->act_x.max_running = xmax;
    if (xmin < g->act_x.min_running) g->act_x.min_running = xmin;
    g->act_x.samples++;
    const float hmin = -(float)q_h.zp * q_h.scale;
    const float hmax = (255.0f - (float)q_h.zp) * q_h.scale;
    if (hmax > g->act_h.max_running) g->act_h.max_running = hmax;
    if (hmin < g->act_h.min_running) g->act_h.min_running = hmin;
    g->act_h.samples++;
}

void fe_gru_step_fp16h(FeGRU *g, const float *x,
                        uint16_t *h_inout_fp16, float *h_out_scratch,
                        int freq, int8_t *aq, int32_t *c32) {
    const int D = g->hidden_size;

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if ((fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM
         || fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD
         || fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_NEON)
        && (D & 7) == 0 && (freq & 7) == 0) {
        const int tier = fe_qgemm_ops.tier;
        FE_TIME_BEGIN("gru_full_fused_arm");
        FeGruFp16Setup s = fe_gru_fp16_setup(g, x, h_inout_fp16, freq, aq);
        if (tier == FE_QGEMM_TIER_ARM_I8MM) {
            qgemm_i8mm_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        } else if (tier == FE_QGEMM_TIER_ARM_DOTPROD) {
            qgemm_dotprod_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        } else {
            qgemm_neon_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        }
        fe_gru_fp16_track_act(g, s.q_x, s.q_h);
        FE_TIME_END();
        (void)c32;
        return;
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if ((fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI
         || fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI
         || fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX2)
        && (D & 7) == 0 && (freq & 7) == 0) {
        const int tier = fe_qgemm_ops.tier;
        FE_TIME_BEGIN("gru_full_fused_x86");
        FeGruFp16Setup s = fe_gru_fp16_setup(g, x, h_inout_fp16, freq, aq);
        if (tier == FE_QGEMM_TIER_X86_AVX512_VNNI) {
            qgemm_avx512vnni_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        } else if (tier == FE_QGEMM_TIER_X86_AVX_VNNI) {
            qgemm_avxvnni_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        } else {
            qgemm_avx2_gru_full_fused_fp16inout(
                freq, D, s.x_q, D, s.h_q, D,
                g->Wq_ih, g->Wq_hh,
                g_gru_combined_ih, g_gru_bias_eff_ih,
                g_gru_combined_hh, g_gru_bias_eff_hh,
                s.br_sum, s.bz_sum, s.bn_i, s.bn_h,
                h_inout_fp16, D, h_out_scratch, D);
        }
        fe_gru_fp16_track_act(g, s.q_x, s.q_h);
        FE_TIME_END();
        (void)c32;
        return;
    }
#endif

    /* No SIMD tier => unsupported ISA. The engine guarantees a SIMD tier on
     * every supported target (aarch64 NEON; x86 AVX2+FMA3+F16C enforced at
     * fe_qgemm_init), so this is unreachable. Abort loudly rather than run
     * a scalar path — unsupported ISAs do not run (project policy). */
    (void)g; (void)x; (void)h_inout_fp16; (void)freq; (void)aq; (void)c32;
    (void)h_out_scratch;
    abort();
}

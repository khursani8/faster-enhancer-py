/* Engine-internal prototypes shared across the TUs. */
#ifndef FE_INTERNAL_H
#define FE_INTERNAL_H

#include <stdint.h>
#include "fe_config_medium.h"
#include "fe_weights.h"
#include "fe_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Engine ---- */
FeState *fe_state_create(void);
void     fe_state_destroy(FeState *s);
void     fe_process_frame(FeState *s, FeWeights *w,
                          const float *audio_in, float *audio_out);

/* 1024-pt real FFT (src/fft/). */
void fe_fft_init(void);
void fe_rfft (const float *in, float *re, float *im);
void fe_irfft(const float *re, const float *im, float *out);

/* ---- STFT ---- */
void fe_stft_init(FeState *s);
void fe_stft (FeState *s, const float *audio_in);   /* writes spec_cf (split, compressed) */
void fe_istft(FeState *s, float *audio_out);        /* reads  spec_out (compressed) */

/* Conv1d. Non-const FeConv1d so running diagnostics may update it. */
void fe_conv1d_k1_FCin(FeConv1d *c, const float *in_FCin, float *out, int freq,
                       int8_t *aq, int32_t *c32);
/* fp16 B-operand concat2. */
void fe_conv1d_k1_silu_concat2_fp16b(FeConv1d *c, const float *in_a,
                                      const uint16_t *in_b_fp16,
                                      float *out, int freq,
                                      int8_t *aq, int32_t *c32);
void fe_conv1d_k3_buf_silu(FeConv1d *c, const float *in, float *out,
                           int freq,
                           int8_t *aq, int32_t *c32);
/* fp16 skip-write variant. skip_out is uint16_t* (IEEE
 * binary16). Eliminates the engine-side scratch + pack hop. */
void fe_conv1d_k3_buf_silu_skip_fp16(FeConv1d *c, const float *in,
                                      float *out, uint16_t *skip_out,
                                      int freq, int8_t *aq, int32_t *c32);

/* Winograd F(2,3) k=3 conv (uses c->wino_weight_q/scales[]). */
void fe_conv1d_k3_winograd_silu(FeConv1d *c, const float *in, float *out,
                                 int freq, int8_t *aq, int32_t *c32);
/* fp16 skip-write Winograd epilogue. */
void fe_conv1d_k3_winograd_silu_skip_fp16(FeConv1d *c, const float *in,
                                           float *out, uint16_t *skip_out,
                                           int freq, int8_t *aq, int32_t *c32);
/* Derive the 4-component Winograd weights from a packed int8 weight +
 * scales. scratch_fp32 must hold at least Co*Ci*3 + 4*Co*Ci floats. */
void fe_winograd_f23_derive_weights(
        const signed char *Wq_packed, const float *scales_w,
        int Co, int Ci,
        signed char *wg_packed_out[4], float *wg_scales_out[4],
        float *scratch_fp32);

void fe_strided_conv1d(const FeConv1d *c, const float *in, float *out,
                       int in_freq, int stride);
void fe_conv_transpose1d(const FeConvT1d *c, const float *in, float *out,
                         int in_freq);

/* ---- GRU step ---- */
/* fp16 inout. h_inout_fp16 is fp16 storage (read for
 * h_old + written with new state via dual-store ngate epilogue);
 * h_out_scratch is fp32 scratch that rnn_fc consumes. All 6 SIMD tiers
 * implement the dual-store path; no scalar fallback (unsupported ISAs make
 * fe_init return non-zero). */
void fe_gru_step_fp16h(FeGRU *g, const float *x,
                        uint16_t *h_inout_fp16, float *h_out_scratch,
                        int freq, int8_t *aq, int32_t *c32);

/* ---- Attention ---- */
void fe_mhsa(FeAttention *a, FeLinear *fc,
             const float *in, float *out,
             float *qkv_buf, float *score_buf, float *attn_buf,
             int8_t *qkv_q, int8_t *Qq, int8_t *Kpq, int8_t *Vpq,
             int8_t *scoresq,
             int freq,
             int8_t *aq, int32_t *c32);

/* Winograd scratch wire-up, called once at fe_state_create. */
void fe_winograd_set_scratch(int32_t *u, int8_t *dq);

/* Activations. */
/* SiLU + fp16 skip-write. buf stays fp32 in-place, skip_out is the
 * fp16-storage enc_skip slot (uint16 holding IEEE binary16). */
void fe_silu_skip_fp16(float *buf, int n, uint16_t *skip_out);
/* Dequant + softmax + int8 quantize from int32 in one pass. Used by the
 * attention Q@K^T post path where combined_scale is uniform. scratch_fp32
 * is one row of fp32 scratch (reuses score_buf). */
void fe_softmax_rows_quant_from_int32(const int32_t *c32, int rows, int cols,
                                       float combined_scale,
                                       float *scratch_fp32,
                                       int8_t *out_q, float inv_scale);
/* Vector ops. */
void fe_vec_add (float *dst, const float *src, int n);

#ifdef __cplusplus
}
#endif

#endif /* FE_INTERNAL_H */

/*
 * Conv entry wrappers: k=1 QGEMM, k=3 Winograd, StridedConv1d
 * (k=8 stride 4), ConvTranspose1d.
 * Single-time-step convs along the frequency axis on [C, F] storage.
 * BatchNorm is pre-fused into weights at export time.
 */
#include "fe_internal.h"
#include "fe_sgemm.h"
#include "fe_qgemm.h"
#include "fe_simd.h"
#include <assert.h>

/* Broadcast b across n floats. */
static inline void fe_broadcast(float *dst, float b, int n) {
    fe_f32x4 vb = fe_set1(b);
    int f = 0;
    for (; f + 15 < n; f += 16) {
        fe_store(dst + f + 0,  vb);
        fe_store(dst + f + 4,  vb);
        fe_store(dst + f + 8,  vb);
        fe_store(dst + f + 12, vb);
    }
    for (; f + 3 < n; f += 4) fe_store(dst + f, vb);
    for (; f < n; ++f) dst[f] = b;
}

/* k=1 conv with input in [C, F] layout; the [C, F] -> [F, C] transpose
 * folds into the int8 quantize pass. */
void fe_conv1d_k1_FCin(FeConv1d *c, const float *in_FCin, float *out, int freq,
                       int8_t *aq, int32_t *c32) {
    fe_qgemm_packed_calib_transposed_in(
        freq, c->out_ch, c->in_ch, in_FCin,
        c->weight_q, c->scales_w, c->row_sums, c->bias,
        out, c->out_ch, aq, c32, &c->act);
}

/* fp16 B-operand variant. in_b is uint16_t* (IEEE binary16).
 * Eliminates the engine-side fp16->fp32 unpack pass before this kernel. */
void fe_conv1d_k1_silu_concat2_fp16b(FeConv1d *c, const float *in_a,
                                      const uint16_t *in_b_fp16,
                                      float *out, int freq,
                                      int8_t *aq, int32_t *c32) {
    const int C_half = c->in_ch / 2;
    fe_qgemm_packed_silu_calib_concat2_fp16b(freq, c->out_ch, C_half,
                                              in_a, in_b_fp16,
                                              c->weight_q, c->scales_w,
                                              c->row_sums, c->bias,
                                              out, c->out_ch, aq, c32,
                                              &c->act);
}

/*
 * k=3 conv is Winograd F(2,3). Weights are derived once at load time
 * (G*W*G^T + symmetric int8 quant + DOTPROD pack) into wino_weight_q[0..3]
 * + wino_scales[0..3]. fe_load_weights fails if this derivation cannot
 * finish, so the production k=3 path is Winograd-only.
 */
void fe_conv1d_k3_buf_silu(FeConv1d *c, const float *in, float *out,
                           int freq,
                           int8_t *aq, int32_t *c32) {
    assert(c->wino_weight_q[0]);
    fe_conv1d_k3_winograd_silu(c, in, out, freq, aq, c32);
}

/* k=3 conv + SiLU + fp16 skip-write. Direct fp16 store at the Winograd
 * epilogue eliminates the engine-side scratch + pack pass. */
void fe_conv1d_k3_buf_silu_skip_fp16(FeConv1d *c, const float *in,
                                      float *out, uint16_t *skip_out,
                                      int freq, int8_t *aq, int32_t *c32) {
    assert(c->wino_weight_q[0]);
    fe_conv1d_k3_winograd_silu_skip_fp16(c, in, out, skip_out, freq, aq, c32);
}

/*
 * StridedConv1d: [2, FE_FREQ_BINS] -> [C1, FE_F1].
 * PyTorch: Conv1d(2->C1, k=8, stride=4, pad=2). Reshape to
 * [2*stride, (F+2*pad)/stride] and run plain Conv1d(8->C1, k=2).
 */
void fe_strided_conv1d(const FeConv1d *c, const float *in, float *out,
                       int in_freq, int stride) {
    const int Ci_orig = c->in_ch / stride;
    const int Co      = c->out_ch;
    const int K       = c->kernel;
    const int pad     = (FE_ENC_K0 - stride) / 2;

    const int F_padded = in_freq + 2 * pad;
    const int Ci       = Ci_orig * stride;
    const int F_new    = F_padded / stride;

    /* Reshaped input [Ci, F_new] -- head-pad / valid range / tail-pad
     * per row, every slot written once. */
    float reshaped[FE_STRIDED_CI * FE_STRIDED_FNEW];
    for (int i = 0; i < Ci; ++i) {
        const int ch    = i % Ci_orig;
        const int phase = i / Ci_orig;
        const float *in_ch = in + (size_t)ch * in_freq;
        float *out_row = reshaped + (size_t)i * F_new;
        int f_start = (pad - phase + stride - 1) / stride;
        if (f_start < 0) f_start = 0;
        if (f_start > F_new) f_start = F_new;
        int f_end_safe = (in_freq - 1 + pad - phase) / stride;
        if (f_end_safe >= F_new) f_end_safe = F_new - 1;
        if (f_end_safe < f_start - 1) f_end_safe = f_start - 1;
        for (int f = 0; f < f_start; ++f) out_row[f] = 0.0f;
        for (int f = f_start; f <= f_end_safe; ++f)
            out_row[f] = in_ch[f * stride + phase - pad];
        for (int f = f_end_safe + 1; f < F_new; ++f) out_row[f] = 0.0f;
    }

    int out_freq = F_new - K + 1;

    /* k=2 conv: weight [Co, Ci, 2] viewed as [Co, 2*Ci] row-major
     * (zero-copy); im2col_T [out_freq, 2*Ci]; out = im2col_T @ W^T. */
    float im2col_T[FE_F1 * 2 * FE_STRIDED_CI];
    for (int f = 0; f < out_freq; ++f) {
        float *row = im2col_T + (size_t)f * (2 * Ci);
        for (int i = 0; i < Ci; ++i) {
            const float *src = reshaped + (size_t)i * F_new + f;
            row[i * 2 + 0] = src[0];
            row[i * 2 + 1] = src[1];
        }
    }

    /* Pre-packed weight from load time. */
    fe_linear_packed(out_freq, Co, 2 * Ci, im2col_T,
                     c->weight_packed_fp32, c->bias, out);
}

/* ConvTranspose1d: [F, Ci] -> [Co, out_freq]. Weight [Ci, Co, K]. */
void fe_conv_transpose1d(const FeConvT1d *c, const float *in, float *out,
                         int in_freq) {
    const int Ci  = c->in_ch;
    const int Co  = c->out_ch;
    const int K   = c->kernel;
    const int S   = c->stride;
    const int pad = (K - S) / 2;
    const int out_freq = (in_freq - 1) * S - 2 * pad + K;

    /* Zero + bias broadcast */
    for (int o = 0; o < Co; ++o)
        fe_broadcast(out + (size_t)o * out_freq, c->bias[o], out_freq);

    /* Range where all K taps fall inside the output. */
    const int f_start = (pad + S - 1) / S;
    const int f_end   = (out_freq - K + pad) / S;

    /* K=8 in this engine -> exactly 2 fe_f32x4 stamps per tap. Edges
     * use a scalar guard; interior uses unmasked vector stores. Loop
     * order (i, f) outer with Co=2 unrolled hoists the in[] load. */
    if (Co == 2) {
        for (int i = 0; i < Ci; ++i) {
            const float *ww0      = c->weight + ((size_t)i * Co + 0) * K;
            const float *ww1      = c->weight + ((size_t)i * Co + 1) * K;
            float       *out_row0 = out + (size_t)0 * out_freq;
            float       *out_row1 = out + (size_t)1 * out_freq;
            fe_f32x4 w0_0 = fe_load(ww0 + 0), w0_1 = fe_load(ww0 + 4);
            fe_f32x4 w1_0 = fe_load(ww1 + 0), w1_1 = fe_load(ww1 + 4);

            for (int f = 0; f < f_start; ++f) {
                float x = in[(size_t)f * Ci + i];
                int base = f * S - pad;
                for (int k = 0; k < K; ++k) {
                    int p = base + k;
                    if (p >= 0 && p < out_freq) {
                        out_row0[p] += ww0[k] * x;
                        out_row1[p] += ww1[k] * x;
                    }
                }
            }
            for (int f = f_start; f <= f_end; ++f) {
                fe_f32x4 vx = fe_set1(in[(size_t)f * Ci + i]);
                int base = f * S - pad;
                fe_store(out_row0 + base + 0, fe_fma(w0_0, vx, fe_load(out_row0 + base + 0)));
                fe_store(out_row0 + base + 4, fe_fma(w0_1, vx, fe_load(out_row0 + base + 4)));
                fe_store(out_row1 + base + 0, fe_fma(w1_0, vx, fe_load(out_row1 + base + 0)));
                fe_store(out_row1 + base + 4, fe_fma(w1_1, vx, fe_load(out_row1 + base + 4)));
            }
            for (int f = f_end + 1; f < in_freq; ++f) {
                float x = in[(size_t)f * Ci + i];
                int base = f * S - pad;
                for (int k = 0; k < K; ++k) {
                    int p = base + k;
                    if (p >= 0 && p < out_freq) {
                        out_row0[p] += ww0[k] * x;
                        out_row1[p] += ww1[k] * x;
                    }
                }
            }
        }
    } else {
        /* Generic Co fallback. */
        for (int i = 0; i < Ci; ++i) {
            for (int o = 0; o < Co; ++o) {
                const float *ww      = c->weight + ((size_t)i * Co + o) * K;
                float       *out_row = out + (size_t)o * out_freq;
                fe_f32x4 w0 = fe_load(ww + 0);
                fe_f32x4 w1 = fe_load(ww + 4);

                for (int f = 0; f < f_start; ++f) {
                    float x = in[(size_t)f * Ci + i];
                    int base = f * S - pad;
                    for (int k = 0; k < K; ++k) {
                        int p = base + k;
                        if (p >= 0 && p < out_freq) out_row[p] += ww[k] * x;
                    }
                }
                for (int f = f_start; f <= f_end; ++f) {
                    fe_f32x4 vx = fe_set1(in[(size_t)f * Ci + i]);
                    int base = f * S - pad;
                    fe_store(out_row + base + 0, fe_fma(w0, vx, fe_load(out_row + base + 0)));
                    fe_store(out_row + base + 4, fe_fma(w1, vx, fe_load(out_row + base + 4)));
                }
                for (int f = f_end + 1; f < in_freq; ++f) {
                    float x = in[(size_t)f * Ci + i];
                    int base = f * S - pad;
                    for (int k = 0; k < K; ++k) {
                        int p = base + k;
                        if (p >= 0 && p < out_freq) out_row[p] += ww[k] * x;
                    }
                }
            }
        }
    }
}

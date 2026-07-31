/* NEON 4-wide 1024-pt real FFT. Stockham radix-4 (L=1..64), final
 * radix-2 (L=256), then real-FFT pre/post. Stage 0 uses vst4q_f32 to
 * scatter 4 butterflies into 16 contiguous outputs. */
#include <arm_neon.h>
#include <math.h>
#include <string.h>

/* Complex multiply, 4-wide. */
static inline void fe_neon_cmul(float32x4_t a_re, float32x4_t a_im,
                                float32x4_t b_re, float32x4_t b_im,
                                float32x4_t *out_re, float32x4_t *out_im) {
    /* (ar + i ai)(br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br). */
    float32x4_t r = vmulq_f32(a_re, b_re);
    r = vfmsq_f32(r, a_im, b_im);
    float32x4_t i = vmulq_f32(a_re, b_im);
    i = vfmaq_f32(i, a_im, b_re);
    *out_re = r;
    *out_im = i;
}

/* Radix-4 butterfly (4-wide). Caller pre-multiplies a1..a3 by twiddles. */
static inline void fe_neon_radix4_butterfly(
        float32x4_t a0_re, float32x4_t a0_im,
        float32x4_t b1_re, float32x4_t b1_im,
        float32x4_t b2_re, float32x4_t b2_im,
        float32x4_t b3_re, float32x4_t b3_im,
        float32x4_t *o0_re, float32x4_t *o0_im,
        float32x4_t *o1_re, float32x4_t *o1_im,
        float32x4_t *o2_re, float32x4_t *o2_im,
        float32x4_t *o3_re, float32x4_t *o3_im) {
    float32x4_t t0_re = vaddq_f32(a0_re, b2_re), t0_im = vaddq_f32(a0_im, b2_im);
    float32x4_t t1_re = vsubq_f32(a0_re, b2_re), t1_im = vsubq_f32(a0_im, b2_im);
    float32x4_t t2_re = vaddq_f32(b1_re, b3_re), t2_im = vaddq_f32(b1_im, b3_im);
    float32x4_t t3_re = vsubq_f32(b1_re, b3_re), t3_im = vsubq_f32(b1_im, b3_im);

    *o0_re = vaddq_f32(t0_re, t2_re); *o0_im = vaddq_f32(t0_im, t2_im);
    *o1_re = vaddq_f32(t1_re, t3_im); *o1_im = vsubq_f32(t1_im, t3_re);
    *o2_re = vsubq_f32(t0_re, t2_re); *o2_im = vsubq_f32(t0_im, t2_im);
    *o3_re = vsubq_f32(t1_re, t3_im); *o3_im = vaddq_f32(t1_im, t3_re);
}

/* Stage 0 (L=1): no twiddles. 4 work units/iter via vst4q_f32. */
static void fft_neon_stage0(const float *in_re, const float *in_im,
                            float *out_re, float *out_im) {
    const int stride = FE_FFT_HALF / 4;
    for (int j = 0; j < stride; j += 4) {
        float32x4_t a0_re = vld1q_f32(in_re + j + 0 * stride);
        float32x4_t a0_im = vld1q_f32(in_im + j + 0 * stride);
        float32x4_t a1_re = vld1q_f32(in_re + j + 1 * stride);
        float32x4_t a1_im = vld1q_f32(in_im + j + 1 * stride);
        float32x4_t a2_re = vld1q_f32(in_re + j + 2 * stride);
        float32x4_t a2_im = vld1q_f32(in_im + j + 2 * stride);
        float32x4_t a3_re = vld1q_f32(in_re + j + 3 * stride);
        float32x4_t a3_im = vld1q_f32(in_im + j + 3 * stride);

        float32x4_t o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
        fe_neon_radix4_butterfly(a0_re, a0_im, a1_re, a1_im, a2_re, a2_im, a3_re, a3_im,
                                  &o0_re, &o0_im, &o1_re, &o1_im,
                                  &o2_re, &o2_im, &o3_re, &o3_im);

        /* vst4q_f32 lays out 4 vectors as [v0.0,v1.0,v2.0,v3.0,
         * v0.1,...] -- exactly the per-butterfly contiguous outputs. */
        float32x4x4_t pack_re = { { o0_re, o1_re, o2_re, o3_re } };
        float32x4x4_t pack_im = { { o0_im, o1_im, o2_im, o3_im } };
        vst4q_f32(out_re + 4 * j, pack_re);
        vst4q_f32(out_im + 4 * j, pack_im);
    }
}

/* Generic radix-4 stage (L >= 4). */
static void fft_neon_radix4_stage(const float *in_re, const float *in_im,
                                  float *out_re, float *out_im,
                                  const float *twr, const float *twi,
                                  int L) {
    const int Ls = L * 4;
    const int n_groups = FE_FFT_HALF / Ls;
    const int stride = FE_FFT_HALF / 4;
    for (int j = 0; j < n_groups; ++j) {
        for (int k = 0; k < L; k += 4) {
            const int in_base = j * L + k;
            float32x4_t a0_re = vld1q_f32(in_re + in_base + 0 * stride);
            float32x4_t a0_im = vld1q_f32(in_im + in_base + 0 * stride);
            float32x4_t a1_re = vld1q_f32(in_re + in_base + 1 * stride);
            float32x4_t a1_im = vld1q_f32(in_im + in_base + 1 * stride);
            float32x4_t a2_re = vld1q_f32(in_re + in_base + 2 * stride);
            float32x4_t a2_im = vld1q_f32(in_im + in_base + 2 * stride);
            float32x4_t a3_re = vld1q_f32(in_re + in_base + 3 * stride);
            float32x4_t a3_im = vld1q_f32(in_im + in_base + 3 * stride);

            float32x4_t w1_re = vld1q_f32(twr + 0 * L + k);
            float32x4_t w1_im = vld1q_f32(twi + 0 * L + k);
            float32x4_t w2_re = vld1q_f32(twr + 1 * L + k);
            float32x4_t w2_im = vld1q_f32(twi + 1 * L + k);
            float32x4_t w3_re = vld1q_f32(twr + 2 * L + k);
            float32x4_t w3_im = vld1q_f32(twi + 2 * L + k);

            float32x4_t b1_re, b1_im, b2_re, b2_im, b3_re, b3_im;
            fe_neon_cmul(a1_re, a1_im, w1_re, w1_im, &b1_re, &b1_im);
            fe_neon_cmul(a2_re, a2_im, w2_re, w2_im, &b2_re, &b2_im);
            fe_neon_cmul(a3_re, a3_im, w3_re, w3_im, &b3_re, &b3_im);

            float32x4_t o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
            fe_neon_radix4_butterfly(a0_re, a0_im, b1_re, b1_im, b2_re, b2_im, b3_re, b3_im,
                                      &o0_re, &o0_im, &o1_re, &o1_im,
                                      &o2_re, &o2_im, &o3_re, &o3_im);

            const int out_base = j * Ls + k;
            vst1q_f32(out_re + out_base + 0 * L, o0_re);
            vst1q_f32(out_im + out_base + 0 * L, o0_im);
            vst1q_f32(out_re + out_base + 1 * L, o1_re);
            vst1q_f32(out_im + out_base + 1 * L, o1_im);
            vst1q_f32(out_re + out_base + 2 * L, o2_re);
            vst1q_f32(out_im + out_base + 2 * L, o2_im);
            vst1q_f32(out_re + out_base + 3 * L, o3_re);
            vst1q_f32(out_im + out_base + 3 * L, o3_im);
        }
    }
}

/* Final radix-2 stage (L=256). */
static void fft_neon_radix2_stage(const float *in_re, const float *in_im,
                                  float *out_re, float *out_im,
                                  const float *twr, const float *twi) {
    const int L = 256;
    for (int k = 0; k < L; k += 4) {
        float32x4_t a_re = vld1q_f32(in_re + k);
        float32x4_t a_im = vld1q_f32(in_im + k);
        float32x4_t b_re = vld1q_f32(in_re + k + L);
        float32x4_t b_im = vld1q_f32(in_im + k + L);
        float32x4_t w_re = vld1q_f32(twr + k);
        float32x4_t w_im = vld1q_f32(twi + k);
        float32x4_t bw_re, bw_im;
        fe_neon_cmul(b_re, b_im, w_re, w_im, &bw_re, &bw_im);
        vst1q_f32(out_re + k,     vaddq_f32(a_re, bw_re));
        vst1q_f32(out_im + k,     vaddq_f32(a_im, bw_im));
        vst1q_f32(out_re + k + L, vsubq_f32(a_re, bw_re));
        vst1q_f32(out_im + k + L, vsubq_f32(a_im, bw_im));
    }
}

/* 512-pt complex FFT (NEON). */
static void fft_complex_512_neon(const float *in_re, const float *in_im,
                                 float *out_re, float *out_im) {
    /* Ping-pong scratch. */
    static __thread float buf_re[2][FE_FFT_HALF];
    static __thread float buf_im[2][FE_FFT_HALF];

    fft_neon_stage0(in_re, in_im, buf_re[0], buf_im[0]);

    int parity = 0;
    int L = 4;
    for (int s = 1; s < FE_FFT_R4_STAGES; ++s) {
        const float *twr = g_fft_plan.twiddle_r4_re + g_fft_plan.twiddle_r4_off[s];
        const float *twi = g_fft_plan.twiddle_r4_im + g_fft_plan.twiddle_r4_off[s];
        fft_neon_radix4_stage(buf_re[parity], buf_im[parity],
                              buf_re[1 - parity], buf_im[1 - parity],
                              twr, twi, L);
        parity = 1 - parity;
        L *= 4;
    }
    fft_neon_radix2_stage(buf_re[parity], buf_im[parity], out_re, out_im,
                          g_fft_plan.twiddle_r2_re, g_fft_plan.twiddle_r2_im);
}

/* Pack 1024 real -> 512 complex (interleaved evens/odds). */
static inline void fft_neon_pack_real(const float *in_real,
                                      float *pack_re, float *pack_im) {
    for (int k = 0; k < FE_FFT_HALF; k += 4) {
        /* vld2q_f32: val[0] = evens, val[1] = odds. */
        float32x4x2_t v = vld2q_f32(in_real + 2 * k);
        vst1q_f32(pack_re + k, v.val[0]);
        vst1q_f32(pack_im + k, v.val[1]);
    }
}

/* Real-FFT post-process (forward), k=1..HALF-1. */
static void fft_neon_real_post_forward(const float *Z_re, const float *Z_im,
                                       float *out_re, float *out_im) {
    out_re[0]           = Z_re[0] + Z_im[0]; out_im[0]           = 0.0f;
    out_re[FE_FFT_HALF] = Z_re[0] - Z_im[0]; out_im[FE_FFT_HALF] = 0.0f;

    const float32x4_t half = vdupq_n_f32(0.5f);
    int k = 1;
    for (; k + 3 < FE_FFT_HALF; k += 4) {
        float32x4_t zk_r = vld1q_f32(Z_re + k);
        float32x4_t zk_i = vld1q_f32(Z_im + k);
        float zn_r_tmp[4], zn_i_tmp[4];
        for (int i = 0; i < 4; ++i) {
            zn_r_tmp[i] = Z_re[FE_FFT_HALF - (k + i)];
            zn_i_tmp[i] = Z_im[FE_FFT_HALF - (k + i)];
        }
        float32x4_t zn_r = vld1q_f32(zn_r_tmp);
        float32x4_t zn_i = vld1q_f32(zn_i_tmp);
        float32x4_t sum_r = vmulq_f32(half, vaddq_f32(zk_r, zn_r));
        float32x4_t sum_i = vmulq_f32(half, vsubq_f32(zk_i, zn_i));
        float32x4_t dif_r = vmulq_f32(half, vsubq_f32(zk_r, zn_r));
        float32x4_t dif_i = vmulq_f32(half, vaddq_f32(zk_i, zn_i));
        float32x4_t tw_r = vld1q_f32(g_fft_plan.realfft_re + k);
        float32x4_t tw_i = vld1q_f32(g_fft_plan.realfft_im + k);
        float32x4_t d_re, d_im;
        fe_neon_cmul(dif_r, dif_i, tw_r, tw_i, &d_re, &d_im);
        vst1q_f32(out_re + k, vaddq_f32(sum_r, d_im));
        vst1q_f32(out_im + k, vsubq_f32(sum_i, d_re));
    }
    /* Scalar tail. */
    for (; k < FE_FFT_HALF; ++k) {
        const float zk_r = Z_re[k];
        const float zk_i = Z_im[k];
        const float zn_r = Z_re[FE_FFT_HALF - k];
        const float zn_i = Z_im[FE_FFT_HALF - k];
        const float sum_r = 0.5f * (zk_r + zn_r);
        const float sum_i = 0.5f * (zk_i - zn_i);
        const float dif_r = 0.5f * (zk_r - zn_r);
        const float dif_i = 0.5f * (zk_i + zn_i);
        const float tw_r = g_fft_plan.realfft_re[k];
        const float tw_i = g_fft_plan.realfft_im[k];
        const float d_r = dif_r * tw_r - dif_i * tw_i;
        const float d_i = dif_r * tw_i + dif_i * tw_r;
        out_re[k] = sum_r + d_i;
        out_im[k] = sum_i - d_r;
    }
}

/* Real-FFT pre-process (inverse): rebuild 512 complex from 513 bins. */
static void fft_neon_real_pre_inverse(const float *in_re, const float *in_im,
                                      float *Z_re, float *Z_im) {
    Z_re[0] = in_re[0] + in_re[FE_FFT_HALF];
    Z_im[0] = in_re[0] - in_re[FE_FFT_HALF];

    int k = 1;
    for (; k + 3 < FE_FFT_HALF; k += 4) {
        float32x4_t xk_r = vld1q_f32(in_re + k);
        float32x4_t xk_i = vld1q_f32(in_im + k);
        float xn_r_tmp[4], xn_i_tmp[4];
        for (int i = 0; i < 4; ++i) {
            xn_r_tmp[i] = in_re[FE_FFT_HALF - (k + i)];
            xn_i_tmp[i] = in_im[FE_FFT_HALF - (k + i)];
        }
        float32x4_t xn_r = vld1q_f32(xn_r_tmp);
        float32x4_t xn_i = vld1q_f32(xn_i_tmp);
        float32x4_t sum_r = vaddq_f32(xk_r, xn_r);
        float32x4_t sum_i = vsubq_f32(xk_i, xn_i);
        float32x4_t dif_r = vsubq_f32(xk_r, xn_r);
        float32x4_t dif_i = vaddq_f32(xk_i, xn_i);
        float32x4_t tw_r =        vld1q_f32(g_fft_plan.realfft_re + k);
        float32x4_t tw_i = vnegq_f32(vld1q_f32(g_fft_plan.realfft_im + k));
        float32x4_t d_re, d_im;
        fe_neon_cmul(dif_r, dif_i, tw_r, tw_i, &d_re, &d_im);
        vst1q_f32(Z_re + k, vsubq_f32(sum_r, d_im));
        vst1q_f32(Z_im + k, vaddq_f32(sum_i, d_re));
    }
    for (; k < FE_FFT_HALF; ++k) {
        const float xk_r = in_re[k],                 xk_i = in_im[k];
        const float xn_r = in_re[FE_FFT_HALF - k],   xn_i = in_im[FE_FFT_HALF - k];
        const float sum_r = xk_r + xn_r;
        const float sum_i = xk_i - xn_i;
        const float dif_r = xk_r - xn_r;
        const float dif_i = xk_i + xn_i;
        const float tw_r =  g_fft_plan.realfft_re[k];
        const float tw_i = -g_fft_plan.realfft_im[k];
        const float d_r = dif_r * tw_r - dif_i * tw_i;
        const float d_i = dif_r * tw_i + dif_i * tw_r;
        Z_re[k] = sum_r - d_i;
        Z_im[k] = sum_i + d_r;
    }
}

/* Public entry points. */
void fe_fft_forward_neon(const float *in_real, float *out_re, float *out_im) {
    float pack_re[FE_FFT_HALF], pack_im[FE_FFT_HALF];
    fft_neon_pack_real(in_real, pack_re, pack_im);

    float Z_re[FE_FFT_HALF], Z_im[FE_FFT_HALF];
    fft_complex_512_neon(pack_re, pack_im, Z_re, Z_im);

    fft_neon_real_post_forward(Z_re, Z_im, out_re, out_im);
}

void fe_fft_inverse_neon(const float *in_re, const float *in_im, float *out_real) {
    float Z_re[FE_FFT_HALF], Z_im[FE_FFT_HALF];
    fft_neon_real_pre_inverse(in_re, in_im, Z_re, Z_im);

    /* Inverse complex FFT via conjugation trick: invFFT(X) = conj(FFT(conj(X)))/N.
     * 1/N scaling is folded into window_istft at init time, so we don't apply it
     * here. Conjugate the input, run forward complex FFT, then conjugate output. */
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (int k = 0; k < FE_FFT_HALF; k += 4) {
        float32x4_t z_im = vld1q_f32(Z_im + k);
        vst1q_f32(Z_im + k, vsubq_f32(zero, z_im));
    }

    float W_re[FE_FFT_HALF], W_im[FE_FFT_HALF];
    fft_complex_512_neon(Z_re, Z_im, W_re, W_im);

    /* Unpack: real_out[2k] = W_re[k], real_out[2k+1] = -W_im[k]. Use vst2 for
     * interleaved store.                                                    */
    for (int k = 0; k < FE_FFT_HALF; k += 4) {
        float32x4_t r = vld1q_f32(W_re + k);
        float32x4_t i = vld1q_f32(W_im + k);
        i = vsubq_f32(zero, i);
        float32x4x2_t pack = { { r, i } };
        vst2q_f32(out_real + 2 * k, pack);
    }
}

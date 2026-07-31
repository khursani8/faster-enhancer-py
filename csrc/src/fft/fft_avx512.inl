/* AVX-512 16-wide 1024-pt real FFT. Inner stages with L >= 16 run
 * 16-wide; earlier small-L stages reuse the AVX2 / 128-bit kernels. */
#include <math.h>
#include <string.h>
#include <immintrin.h>

static inline void fe_avx512_cmul(__m512 a_re, __m512 a_im,
                                  __m512 b_re, __m512 b_im,
                                  __m512 *out_re, __m512 *out_im) {
    __m512 r = _mm512_fmsub_ps(a_re, b_re, _mm512_mul_ps(a_im, b_im));
    __m512 i = _mm512_fmadd_ps(a_re, b_im, _mm512_mul_ps(a_im, b_re));
    *out_re = r;
    *out_im = i;
}

static inline void fe_avx512_radix4_butterfly(
        __m512 a0_re, __m512 a0_im,
        __m512 b1_re, __m512 b1_im,
        __m512 b2_re, __m512 b2_im,
        __m512 b3_re, __m512 b3_im,
        __m512 *o0_re, __m512 *o0_im,
        __m512 *o1_re, __m512 *o1_im,
        __m512 *o2_re, __m512 *o2_im,
        __m512 *o3_re, __m512 *o3_im) {
    __m512 t0_re = _mm512_add_ps(a0_re, b2_re), t0_im = _mm512_add_ps(a0_im, b2_im);
    __m512 t1_re = _mm512_sub_ps(a0_re, b2_re), t1_im = _mm512_sub_ps(a0_im, b2_im);
    __m512 t2_re = _mm512_add_ps(b1_re, b3_re), t2_im = _mm512_add_ps(b1_im, b3_im);
    __m512 t3_re = _mm512_sub_ps(b1_re, b3_re), t3_im = _mm512_sub_ps(b1_im, b3_im);

    *o0_re = _mm512_add_ps(t0_re, t2_re); *o0_im = _mm512_add_ps(t0_im, t2_im);
    *o1_re = _mm512_add_ps(t1_re, t3_im); *o1_im = _mm512_sub_ps(t1_im, t3_re);
    *o2_re = _mm512_sub_ps(t0_re, t2_re); *o2_im = _mm512_sub_ps(t0_im, t2_im);
    *o3_re = _mm512_sub_ps(t1_re, t3_im); *o3_im = _mm512_add_ps(t1_im, t3_re);
}

/* 16x4 transpose store: per-lane 4x4 (unpack+shuffle), then
 * permutex2var + shuffle_f32x4 to gather butterfly-major output. */
static inline void fft_avx512_transpose_16x4(
        float *out, __m512 o0, __m512 o1, __m512 o2, __m512 o3) {
    /* Step 1: per-lane 4x4 transpose. */
    __m512 ulo  = _mm512_unpacklo_ps(o0, o1);
    __m512 uhi  = _mm512_unpackhi_ps(o0, o1);
    __m512 ulo2 = _mm512_unpacklo_ps(o2, o3);
    __m512 uhi2 = _mm512_unpackhi_ps(o2, o3);
    __m512 r0 = _mm512_shuffle_ps(ulo,  ulo2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 r1 = _mm512_shuffle_ps(ulo,  ulo2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 r2 = _mm512_shuffle_ps(uhi,  uhi2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 r3 = _mm512_shuffle_ps(uhi,  uhi2, _MM_SHUFFLE(3, 2, 3, 2));

    /* Step 2: combine r0&r1 and r2&r3 across 128-bit lanes. */
    const __m512i idx_lo = _mm512_set_epi32(23, 22, 21, 20,  7,  6,  5,  4,
                                             19, 18, 17, 16,  3,  2,  1,  0);
    const __m512i idx_hi = _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12,
                                             27, 26, 25, 24, 11, 10,  9,  8);
    __m512 tmp01_lo = _mm512_permutex2var_ps(r0, idx_lo, r1);
    __m512 tmp01_hi = _mm512_permutex2var_ps(r0, idx_hi, r1);
    __m512 tmp23_lo = _mm512_permutex2var_ps(r2, idx_lo, r3);
    __m512 tmp23_hi = _mm512_permutex2var_ps(r2, idx_hi, r3);

    /* Step 3: pick 128-bit lanes -> 4 zmm in butterfly-major order. */
    __m512 out0 = _mm512_shuffle_f32x4(tmp01_lo, tmp23_lo, 0x44);
    __m512 out1 = _mm512_shuffle_f32x4(tmp01_lo, tmp23_lo, 0xEE);
    __m512 out2 = _mm512_shuffle_f32x4(tmp01_hi, tmp23_hi, 0x44);
    __m512 out3 = _mm512_shuffle_f32x4(tmp01_hi, tmp23_hi, 0xEE);

    _mm512_storeu_ps(out +  0, out0);
    _mm512_storeu_ps(out + 16, out1);
    _mm512_storeu_ps(out + 32, out2);
    _mm512_storeu_ps(out + 48, out3);
}

static void fft_avx512_stage0(const float *in_re, const float *in_im,
                              float *out_re, float *out_im) {
    const int stride = FE_FFT_HALF / 4;
    for (int j = 0; j < stride; j += 16) {
        __m512 a0_re = _mm512_loadu_ps(in_re + j + 0 * stride);
        __m512 a0_im = _mm512_loadu_ps(in_im + j + 0 * stride);
        __m512 a1_re = _mm512_loadu_ps(in_re + j + 1 * stride);
        __m512 a1_im = _mm512_loadu_ps(in_im + j + 1 * stride);
        __m512 a2_re = _mm512_loadu_ps(in_re + j + 2 * stride);
        __m512 a2_im = _mm512_loadu_ps(in_im + j + 2 * stride);
        __m512 a3_re = _mm512_loadu_ps(in_re + j + 3 * stride);
        __m512 a3_im = _mm512_loadu_ps(in_im + j + 3 * stride);

        __m512 o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
        fe_avx512_radix4_butterfly(a0_re, a0_im, a1_re, a1_im, a2_re, a2_im, a3_re, a3_im,
                                    &o0_re, &o0_im, &o1_re, &o1_im,
                                    &o2_re, &o2_im, &o3_re, &o3_im);

        fft_avx512_transpose_16x4(out_re + 4 * j, o0_re, o1_re, o2_re, o3_re);
        fft_avx512_transpose_16x4(out_im + 4 * j, o0_im, o1_im, o2_im, o3_im);
    }
}

static void fft_avx512_radix4_stage_wide(const float *in_re, const float *in_im,
                                         float *out_re, float *out_im,
                                         const float *twr, const float *twi,
                                         int L) {
    const int Ls = L * 4;
    const int n_groups = FE_FFT_HALF / Ls;
    const int stride = FE_FFT_HALF / 4;
    for (int j = 0; j < n_groups; ++j) {
        for (int k = 0; k < L; k += 16) {
            const int in_base = j * L + k;
            __m512 a0_re = _mm512_loadu_ps(in_re + in_base + 0 * stride);
            __m512 a0_im = _mm512_loadu_ps(in_im + in_base + 0 * stride);
            __m512 a1_re = _mm512_loadu_ps(in_re + in_base + 1 * stride);
            __m512 a1_im = _mm512_loadu_ps(in_im + in_base + 1 * stride);
            __m512 a2_re = _mm512_loadu_ps(in_re + in_base + 2 * stride);
            __m512 a2_im = _mm512_loadu_ps(in_im + in_base + 2 * stride);
            __m512 a3_re = _mm512_loadu_ps(in_re + in_base + 3 * stride);
            __m512 a3_im = _mm512_loadu_ps(in_im + in_base + 3 * stride);

            __m512 w1_re = _mm512_loadu_ps(twr + 0 * L + k);
            __m512 w1_im = _mm512_loadu_ps(twi + 0 * L + k);
            __m512 w2_re = _mm512_loadu_ps(twr + 1 * L + k);
            __m512 w2_im = _mm512_loadu_ps(twi + 1 * L + k);
            __m512 w3_re = _mm512_loadu_ps(twr + 2 * L + k);
            __m512 w3_im = _mm512_loadu_ps(twi + 2 * L + k);

            __m512 b1_re, b1_im, b2_re, b2_im, b3_re, b3_im;
            fe_avx512_cmul(a1_re, a1_im, w1_re, w1_im, &b1_re, &b1_im);
            fe_avx512_cmul(a2_re, a2_im, w2_re, w2_im, &b2_re, &b2_im);
            fe_avx512_cmul(a3_re, a3_im, w3_re, w3_im, &b3_re, &b3_im);

            __m512 o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
            fe_avx512_radix4_butterfly(a0_re, a0_im, b1_re, b1_im, b2_re, b2_im, b3_re, b3_im,
                                        &o0_re, &o0_im, &o1_re, &o1_im,
                                        &o2_re, &o2_im, &o3_re, &o3_im);

            const int out_base = j * Ls + k;
            _mm512_storeu_ps(out_re + out_base + 0 * L, o0_re);
            _mm512_storeu_ps(out_im + out_base + 0 * L, o0_im);
            _mm512_storeu_ps(out_re + out_base + 1 * L, o1_re);
            _mm512_storeu_ps(out_im + out_base + 1 * L, o1_im);
            _mm512_storeu_ps(out_re + out_base + 2 * L, o2_re);
            _mm512_storeu_ps(out_im + out_base + 2 * L, o2_im);
            _mm512_storeu_ps(out_re + out_base + 3 * L, o3_re);
            _mm512_storeu_ps(out_im + out_base + 3 * L, o3_im);
        }
    }
}

static void fft_avx512_radix2_stage(const float *in_re, const float *in_im,
                                    float *out_re, float *out_im,
                                    const float *twr, const float *twi) {
    const int L = 256;
    for (int k = 0; k < L; k += 16) {
        __m512 a_re = _mm512_loadu_ps(in_re + k);
        __m512 a_im = _mm512_loadu_ps(in_im + k);
        __m512 b_re = _mm512_loadu_ps(in_re + k + L);
        __m512 b_im = _mm512_loadu_ps(in_im + k + L);
        __m512 w_re = _mm512_loadu_ps(twr + k);
        __m512 w_im = _mm512_loadu_ps(twi + k);
        __m512 bw_re, bw_im;
        fe_avx512_cmul(b_re, b_im, w_re, w_im, &bw_re, &bw_im);
        _mm512_storeu_ps(out_re + k,     _mm512_add_ps(a_re, bw_re));
        _mm512_storeu_ps(out_im + k,     _mm512_add_ps(a_im, bw_im));
        _mm512_storeu_ps(out_re + k + L, _mm512_sub_ps(a_re, bw_re));
        _mm512_storeu_ps(out_im + k + L, _mm512_sub_ps(a_im, bw_im));
    }
}

static void fft_complex_512_avx512(const float *in_re, const float *in_im,
                                   float *out_re, float *out_im) {
    static __thread float buf_re[2][FE_FFT_HALF];
    static __thread float buf_im[2][FE_FFT_HALF];

    /* Stage 0 (L=1, 128 work units): 16-wide butterflies. */
    fft_avx512_stage0(in_re, in_im, buf_re[0], buf_im[0]);

    int parity = 0;
    int L = 4;
    for (int s = 1; s < FE_FFT_R4_STAGES; ++s) {
        const float *twr = g_fft_plan.twiddle_r4_re + g_fft_plan.twiddle_r4_off[s];
        const float *twi = g_fft_plan.twiddle_r4_im + g_fft_plan.twiddle_r4_off[s];
        if (L >= 16) {
            fft_avx512_radix4_stage_wide(buf_re[parity], buf_im[parity],
                                         buf_re[1 - parity], buf_im[1 - parity],
                                         twr, twi, L);
        } else {
            /* L only ever takes {4, 16, 64} (starts at 4, x4 per stage),
             * so this is the L=4 stage. */
            fft_x86_radix4_stage(buf_re[parity], buf_im[parity],
                                 buf_re[1 - parity], buf_im[1 - parity],
                                 twr, twi, L);
        }
        parity = 1 - parity;
        L *= 4;
    }
    fft_avx512_radix2_stage(buf_re[parity], buf_im[parity], out_re, out_im,
                            g_fft_plan.twiddle_r2_re, g_fft_plan.twiddle_r2_im);
}

static inline void fft_avx512_pack_real(const float *in_real,
                                        float *pack_re, float *pack_im) {
    for (int k = 0; k < FE_FFT_HALF; k += 16) {
        __m512 v0 = _mm512_loadu_ps(in_real + 2 * k +  0);
        __m512 v1 = _mm512_loadu_ps(in_real + 2 * k + 16);
        /* Deinterleave evens/odds across two 16-wide loads. */
        __m512i ev_idx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                          14, 12, 10,  8,  6,  4,  2,  0);
        __m512i od_idx = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                          15, 13, 11,  9,  7,  5,  3,  1);
        __m512 evens = _mm512_permutex2var_ps(v0, ev_idx, v1);
        __m512 odds  = _mm512_permutex2var_ps(v0, od_idx, v1);
        _mm512_storeu_ps(pack_re + k, evens);
        _mm512_storeu_ps(pack_im + k, odds);
    }
}

static void fft_avx512_real_post_forward(const float *Z_re, const float *Z_im,
                                         float *out_re, float *out_im) {
    out_re[0]           = Z_re[0] + Z_im[0]; out_im[0]           = 0.0f;
    out_re[FE_FFT_HALF] = Z_re[0] - Z_im[0]; out_im[FE_FFT_HALF] = 0.0f;

    const __m512 half = _mm512_set1_ps(0.5f);
    int k = 1;
    for (; k + 15 < FE_FFT_HALF; k += 16) {
        __m512 zk_r = _mm512_loadu_ps(Z_re + k);
        __m512 zk_i = _mm512_loadu_ps(Z_im + k);
        float zn_r_tmp[16], zn_i_tmp[16];
        for (int i = 0; i < 16; ++i) {
            zn_r_tmp[i] = Z_re[FE_FFT_HALF - (k + i)];
            zn_i_tmp[i] = Z_im[FE_FFT_HALF - (k + i)];
        }
        __m512 zn_r = _mm512_loadu_ps(zn_r_tmp);
        __m512 zn_i = _mm512_loadu_ps(zn_i_tmp);
        __m512 sum_r = _mm512_mul_ps(half, _mm512_add_ps(zk_r, zn_r));
        __m512 sum_i = _mm512_mul_ps(half, _mm512_sub_ps(zk_i, zn_i));
        __m512 dif_r = _mm512_mul_ps(half, _mm512_sub_ps(zk_r, zn_r));
        __m512 dif_i = _mm512_mul_ps(half, _mm512_add_ps(zk_i, zn_i));
        __m512 tw_r = _mm512_loadu_ps(g_fft_plan.realfft_re + k);
        __m512 tw_i = _mm512_loadu_ps(g_fft_plan.realfft_im + k);
        __m512 d_re, d_im;
        fe_avx512_cmul(dif_r, dif_i, tw_r, tw_i, &d_re, &d_im);
        _mm512_storeu_ps(out_re + k, _mm512_add_ps(sum_r, d_im));
        _mm512_storeu_ps(out_im + k, _mm512_sub_ps(sum_i, d_re));
    }
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

static void fft_avx512_real_pre_inverse(const float *in_re, const float *in_im,
                                        float *Z_re, float *Z_im) {
    Z_re[0] = in_re[0] + in_re[FE_FFT_HALF];
    Z_im[0] = in_re[0] - in_re[FE_FFT_HALF];

    int k = 1;
    for (; k + 15 < FE_FFT_HALF; k += 16) {
        __m512 xk_r = _mm512_loadu_ps(in_re + k);
        __m512 xk_i = _mm512_loadu_ps(in_im + k);
        float xn_r_tmp[16], xn_i_tmp[16];
        for (int i = 0; i < 16; ++i) {
            xn_r_tmp[i] = in_re[FE_FFT_HALF - (k + i)];
            xn_i_tmp[i] = in_im[FE_FFT_HALF - (k + i)];
        }
        __m512 xn_r = _mm512_loadu_ps(xn_r_tmp);
        __m512 xn_i = _mm512_loadu_ps(xn_i_tmp);
        __m512 sum_r = _mm512_add_ps(xk_r, xn_r);
        __m512 sum_i = _mm512_sub_ps(xk_i, xn_i);
        __m512 dif_r = _mm512_sub_ps(xk_r, xn_r);
        __m512 dif_i = _mm512_add_ps(xk_i, xn_i);
        __m512 tw_r = _mm512_loadu_ps(g_fft_plan.realfft_re + k);
        __m512 tw_i = _mm512_sub_ps(_mm512_setzero_ps(),
                                     _mm512_loadu_ps(g_fft_plan.realfft_im + k));
        __m512 d_re, d_im;
        fe_avx512_cmul(dif_r, dif_i, tw_r, tw_i, &d_re, &d_im);
        _mm512_storeu_ps(Z_re + k, _mm512_sub_ps(sum_r, d_im));
        _mm512_storeu_ps(Z_im + k, _mm512_add_ps(sum_i, d_re));
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

void fe_fft_forward_avx512(const float *in_real, float *out_re, float *out_im) {
    float pack_re[FE_FFT_HALF], pack_im[FE_FFT_HALF];
    fft_avx512_pack_real(in_real, pack_re, pack_im);
    float Z_re[FE_FFT_HALF], Z_im[FE_FFT_HALF];
    fft_complex_512_avx512(pack_re, pack_im, Z_re, Z_im);
    fft_avx512_real_post_forward(Z_re, Z_im, out_re, out_im);
}

void fe_fft_inverse_avx512(const float *in_re, const float *in_im, float *out_real) {
    float Z_re[FE_FFT_HALF], Z_im[FE_FFT_HALF];
    fft_avx512_real_pre_inverse(in_re, in_im, Z_re, Z_im);

    __m512 zero = _mm512_setzero_ps();
    for (int k = 0; k < FE_FFT_HALF; k += 16) {
        __m512 z_im = _mm512_loadu_ps(Z_im + k);
        _mm512_storeu_ps(Z_im + k, _mm512_sub_ps(zero, z_im));
    }
    float W_re[FE_FFT_HALF], W_im[FE_FFT_HALF];
    fft_complex_512_avx512(Z_re, Z_im, W_re, W_im);

    /* Interleaved store via permutex2var: lanes alternate re/im. */
    __m512i lo_idx = _mm512_set_epi32(23,  7, 22,  6, 21,  5, 20,  4,
                                       19,  3, 18,  2, 17,  1, 16,  0);
    __m512i hi_idx = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12,
                                       27, 11, 26, 10, 25,  9, 24,  8);
    for (int k = 0; k < FE_FFT_HALF; k += 16) {
        __m512 r = _mm512_loadu_ps(W_re + k);
        __m512 i = _mm512_sub_ps(zero, _mm512_loadu_ps(W_im + k));
        __m512 out0 = _mm512_permutex2var_ps(r, lo_idx, i);
        __m512 out1 = _mm512_permutex2var_ps(r, hi_idx, i);
        _mm512_storeu_ps(out_real + 2 * k +  0, out0);
        _mm512_storeu_ps(out_real + 2 * k + 16, out1);
    }
}

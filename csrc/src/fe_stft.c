/*
 * Streaming STFT/iSTFT (n_fft=win=1024, periodic Hann). Power compression
 * matches the spec-domain ONNX graph: spec *= mag^(c-1) with c = 0.3.
 * fe_istft applies the inverse spec *= mag^(1/c - 1) before OLA.
 */
#include "fe_internal.h"
#include "fe_simd.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_simd_post.inl"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Fold buffer for synthesis-window normalization; 4*N_FFT covers all
 * supported hop sizes with margin. */
#define FE_FOLD_LEN (4 * FE_N_FFT)

void fe_stft_init(FeState *s) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n/N)  (matches torch.hann_window)
     *
     * Computed in double and narrowed once. cosf is not correctly rounded and
     * different libm implementations disagree in the last bit -- measured, the
     * macOS and bionic results differ for hundreds of the 1024 coefficients.
     * That is enough to make two builds of this engine produce different
     * output for the same input, because the window multiplies every frame.
     * The double path agrees bit for bit across both, so byte-identity extends
     * across platforms rather than stopping at the build. Init-time only, so
     * the wider arithmetic costs nothing at run time. */
    for (int n = 0; n < N; ++n) {
        s->window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (double)n / (double)N));
    }

    /* Synthesis (iSTFT) window: w / sum_of_squared_windows in the overlap region. */
    int K          = (N + H - 1) / H;
    int num_frames = 2 * K - 1;

    float win_sq_sum[FE_N_FFT];
    memset(win_sq_sum, 0, sizeof(win_sq_sum));

    float fold[FE_FOLD_LEN];
    memset(fold, 0, sizeof(fold));
    for (int f = 0; f < num_frames; ++f) {
        int off = f * H;
        for (int n = 0; n < N; ++n) {
            fold[off + n] += s->window[n] * s->window[n];
        }
    }
    int start = (K - 1) * H;
    for (int n = 0; n < N; ++n) win_sq_sum[n] = fold[start + n];
    /* Pre-fold the 1/N iFFT scaling into the synthesis window. */
    const float inv_N = 1.0f / (float)N;
    for (int n = 0; n < N; ++n) s->window_istft[n] = (s->window[n] / win_sq_sum[n]) * inv_N;

    memset(s->cache_stft,  0, sizeof(s->cache_stft));
    memset(s->cache_istft, 0, sizeof(s->cache_istft));
}

/* Analysis: H audio samples -> spec_cf [re | im] (split layout). */
void fe_stft(FeState *s, const float *audio_in) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Build windowed frame: [cache | new] */
    memcpy(s->fft_buf,                  s->cache_stft, FE_CACHE_LEN * sizeof(float));
    memcpy(s->fft_buf + FE_CACHE_LEN,   audio_in,      H * sizeof(float));

    /* Slide cache: drop H samples, keep the last FE_CACHE_LEN. */
    memcpy(s->cache_stft, s->fft_buf + H, FE_CACHE_LEN * sizeof(float));

    /* Apply analysis window (N=1024 is a multiple of 4) */
    for (int n = 0; n < N; n += 4) {
        fe_store(s->fft_buf + n, fe_mul(fe_load(s->fft_buf + n), fe_load(s->window + n)));
    }

    /* Real FFT. fft_re/fft_im are stack-local scratch (alive only until
     * the power-compress pass below). */
    float fft_re[FE_SPEC_BINS], fft_im[FE_SPEC_BINS];
    fe_rfft(s->fft_buf, fft_re, fft_im);

    /* Power compression: spec *= mag^(c-1) = exp2(0.5*(c-1)*log2(mag^2)).
     * Written directly in split layout into spec_cf; Nyquist bin dropped
     * (model keeps it zero). */
    const float half_exp = 0.5f * FE_COMPRESS_IN;
    float *out_re = s->spec_cf;
    float *out_im = s->spec_cf + FE_FREQ_BINS;
    int f = 0;
    /* Clamp mag_sq at eps=1e-10 (equiv to ONNX Clip(mag, 1e-5) before
     * Pow(-0.7)) -- never zero quiet-band bins. */
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
    const __m512 v16half = _mm512_set1_ps(half_exp);
    const __m512 v16eps  = _mm512_set1_ps(1e-10f);
    for (; f + 15 < FE_FREQ_BINS; f += 16) {
        __m512 re = _mm512_loadu_ps(fft_re + f);
        __m512 im = _mm512_loadu_ps(fft_im + f);
        __m512 mag_sq = _mm512_fmadd_ps(re, re, _mm512_mul_ps(im, im));
        __m512 safe = _mm512_max_ps(mag_sq, v16eps);
        __m512 factor = fe_qg_exp2f16(_mm512_mul_ps(v16half, fe_qg_log2f16(safe)));
        _mm512_storeu_ps(out_re + f, _mm512_mul_ps(re, factor));
        _mm512_storeu_ps(out_im + f, _mm512_mul_ps(im, factor));
    }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
    const __m256 v8half = _mm256_set1_ps(half_exp);
    const __m256 v8eps  = _mm256_set1_ps(1e-10f);
    for (; f + 7 < FE_FREQ_BINS; f += 8) {
        __m256 re = _mm256_loadu_ps(fft_re + f);
        __m256 im = _mm256_loadu_ps(fft_im + f);
        __m256 mag_sq = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));
        __m256 safe = _mm256_max_ps(mag_sq, v8eps);
        __m256 factor = fe_qg_exp2f8(_mm256_mul_ps(v8half, fe_qg_log2f8(safe)));
        _mm256_storeu_ps(out_re + f, _mm256_mul_ps(re, factor));
        _mm256_storeu_ps(out_im + f, _mm256_mul_ps(im, factor));
    }
#endif
    const fe_f32x4 vhalf = fe_set1(half_exp);
    const fe_f32x4 veps  = fe_set1(1e-10f);
    for (; f + 3 < FE_FREQ_BINS; f += 4) {
        fe_f32x4 re = fe_load(fft_re + f);
        fe_f32x4 im = fe_load(fft_im + f);
        fe_f32x4 mag_sq = fe_fma(re, re, fe_mul(im, im));
        fe_f32x4 safe   = fe_max(mag_sq, veps);
        fe_f32x4 factor = FE_EXP2F4(fe_mul(vhalf, FE_LOG2F4(safe)));
        fe_store(out_re + f, fe_mul(re, factor));
        fe_store(out_im + f, fe_mul(im, factor));
    }
    for (; f < FE_FREQ_BINS; ++f) {
        float re = fft_re[f];
        float im = fft_im[f];
        float mag_sq = re * re + im * im;
        if (mag_sq < 1e-10f) mag_sq = 1e-10f;          /* clamp, not zero */
        float factor = exp2f(half_exp * log2f(mag_sq));
        out_re[f] = re * factor;
        out_im[f] = im * factor;
    }
}

/* Synthesis: spec_out (compressed) -> H audio samples. */
void fe_istft(FeState *s, float *audio_out) {
    const int N = FE_N_FFT;
    const int H = FE_HOP_SIZE;

    /* Uncompress: spec *= mag^(1/c - 1) = exp2(0.5*(1/c-1)*log2(mag^2)).
     * spec_out arrives split layout from mask_multiply. */
    /* DELIBERATE ASYMMETRY vs the forward path (fe_stft above), do NOT unify:
     *   forward uses max(mag_sq, 1e-10) -- a CLAMP that never zeroes a bin, so
     *           quiet consonants / low-energy bands survive compression;
     *   inverse uses a MASK mag_sq >= 1e-20 ? factor : 0 -- it hard-ZEROES
     *           true-silence bins on reconstruction.
     * The eps differ on purpose too (1e-10 forward vs 1e-20 inverse): the
     * inverse threshold is well below the forward clamp floor so only genuine
     * silence is killed, not bins the forward pass merely lifted to its floor.
     * Each SIMD/scalar branch below applies factor = mask ? factor : 0. */
    const float half_inv = 0.5f * FE_COMPRESS_OUT;
    float re_full[FE_SPEC_BINS], im_full[FE_SPEC_BINS];
    const float *src_re = s->spec_out;
    const float *src_im = s->spec_out + FE_FREQ_BINS;
    int f = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
    const __m512 v16half = _mm512_set1_ps(half_inv);
    const __m512 v16eps  = _mm512_set1_ps(1e-20f);
    for (; f + 15 < FE_FREQ_BINS; f += 16) {
        __m512 re = _mm512_loadu_ps(src_re + f);
        __m512 im = _mm512_loadu_ps(src_im + f);
        __m512 mag_sq = _mm512_fmadd_ps(re, re, _mm512_mul_ps(im, im));
        __mmask16 mask = _mm512_cmp_ps_mask(mag_sq, v16eps, _CMP_GE_OQ);
        __m512 safe = _mm512_max_ps(mag_sq, v16eps);
        __m512 factor = fe_qg_exp2f16(_mm512_mul_ps(v16half, fe_qg_log2f16(safe)));
        factor = _mm512_maskz_mov_ps(mask, factor);
        _mm512_storeu_ps(re_full + f, _mm512_mul_ps(re, factor));
        _mm512_storeu_ps(im_full + f, _mm512_mul_ps(im, factor));
    }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
    const __m256 v8half = _mm256_set1_ps(half_inv);
    const __m256 v8eps  = _mm256_set1_ps(1e-20f);
    for (; f + 7 < FE_FREQ_BINS; f += 8) {
        __m256 re = _mm256_loadu_ps(src_re + f);
        __m256 im = _mm256_loadu_ps(src_im + f);
        __m256 mag_sq = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));
        __m256 mask = _mm256_cmp_ps(mag_sq, v8eps, _CMP_GE_OQ);
        __m256 safe = _mm256_max_ps(mag_sq, v8eps);
        __m256 factor = fe_qg_exp2f8(_mm256_mul_ps(v8half, fe_qg_log2f8(safe)));
        factor = _mm256_and_ps(mask, factor);
        _mm256_storeu_ps(re_full + f, _mm256_mul_ps(re, factor));
        _mm256_storeu_ps(im_full + f, _mm256_mul_ps(im, factor));
    }
#endif
    const fe_f32x4 vhalf = fe_set1(half_inv);
    const fe_f32x4 veps  = fe_set1(1e-20f);
    for (; f + 3 < FE_FREQ_BINS; f += 4) {
        fe_f32x4 re = fe_load(src_re + f);
        fe_f32x4 im = fe_load(src_im + f);
        fe_f32x4 mag_sq = fe_fma(re, re, fe_mul(im, im));
        fe_f32x4 mask   = fe_ge_mask(mag_sq, veps);
        fe_f32x4 safe   = fe_max(mag_sq, veps);
        fe_f32x4 factor = fe_and_mask(mask, FE_EXP2F4(fe_mul(vhalf, FE_LOG2F4(safe))));
        fe_store(re_full + f, fe_mul(re, factor));
        fe_store(im_full + f, fe_mul(im, factor));
    }
    for (; f < FE_FREQ_BINS; ++f) {
        float re = src_re[f];
        float im = src_im[f];
        float mag_sq = re * re + im * im;
        /* zero true silence (vs forward's clamp-not-zero); see note above. */
        float factor = (mag_sq > 1e-20f) ? exp2f(half_inv * log2f(mag_sq)) : 0.0f;
        re_full[f] = re * factor;
        im_full[f] = im * factor;
    }
    re_full[FE_FREQ_BINS] = 0.0f;   /* Nyquist zeroed */
    im_full[FE_FREQ_BINS] = 0.0f;

    /* iFFT */
    fe_irfft(re_full, im_full, s->fft_buf);

    /* Fused synthesis window * overlap-add. n < CL gets FMA with cache;
     * the tail just multiplies by win. */
    {
        const float *win   = s->window_istft;
        const float *cache = s->cache_istft;
        const int CL = FE_CACHE_LEN;
        int n = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        for (; n + 15 < CL; n += 16) {
            __m512 v = _mm512_loadu_ps(s->fft_buf + n);
            __m512 w = _mm512_loadu_ps(win        + n);
            __m512 c = _mm512_loadu_ps(cache      + n);
            _mm512_storeu_ps(s->fft_buf + n, _mm512_fmadd_ps(v, w, c));
        }
        for (; n + 7 < CL; n += 8) {
            __m256 v = _mm256_loadu_ps(s->fft_buf + n);
            __m256 w = _mm256_loadu_ps(win        + n);
            __m256 c = _mm256_loadu_ps(cache      + n);
            _mm256_storeu_ps(s->fft_buf + n, _mm256_fmadd_ps(v, w, c));
        }
        for (; n + 15 < N; n += 16) {
            __m512 v = _mm512_loadu_ps(s->fft_buf + n);
            __m512 w = _mm512_loadu_ps(win        + n);
            _mm512_storeu_ps(s->fft_buf + n, _mm512_mul_ps(v, w));
        }
        for (; n + 7 < N; n += 8) {
            __m256 v = _mm256_loadu_ps(s->fft_buf + n);
            __m256 w = _mm256_loadu_ps(win        + n);
            _mm256_storeu_ps(s->fft_buf + n, _mm256_mul_ps(v, w));
        }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        for (; n + 7 < CL; n += 8) {
            __m256 v = _mm256_loadu_ps(s->fft_buf + n);
            __m256 w = _mm256_loadu_ps(win        + n);
            __m256 c = _mm256_loadu_ps(cache      + n);
            _mm256_storeu_ps(s->fft_buf + n, _mm256_fmadd_ps(v, w, c));
        }
        for (; n + 7 < N; n += 8) {
            __m256 v = _mm256_loadu_ps(s->fft_buf + n);
            __m256 w = _mm256_loadu_ps(win        + n);
            _mm256_storeu_ps(s->fft_buf + n, _mm256_mul_ps(v, w));
        }
#endif
        for (; n + 3 < CL; n += 4) {
            fe_store(s->fft_buf + n,
                     fe_fma(fe_load(s->fft_buf + n),
                            fe_load(win        + n),
                            fe_load(cache      + n)));
        }
        for (; n + 3 < N; n += 4) {
            fe_store(s->fft_buf + n,
                     fe_mul(fe_load(s->fft_buf + n), fe_load(win + n)));
        }
        for (; n < N; ++n) {
            s->fft_buf[n] = s->fft_buf[n] * win[n] + (n < CL ? cache[n] : 0.0f);
        }
    }

    /* Emit first H samples; shift cache. */
    memcpy(audio_out, s->fft_buf, H * sizeof(float));
    memcpy(s->cache_istft, s->fft_buf + H, FE_CACHE_LEN * sizeof(float));
}

/*
 * 1024-point real FFT/iFFT. Stockham auto-sort radix-4 (4 stages) plus a
 * final radix-2 for the 512-pt complex inner transform, with standard
 * real-FFT pre/post processing. Twiddles precomputed in fe_fft_init.
 */
#include "fe_internal.h"
#include "fe_simd.h"
#include "fft/fft_arch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

/* Plan owned here; .inl kernels reference via the extern in fft_arch.h. */
fe_fft_plan_t g_fft_plan;

static void plan_fill_r4_stage(int s, int L) {
    const int Ls = L * 4;
    const int off = g_fft_plan.twiddle_r4_off[s];
    for (int q = 1; q <= 3; ++q) {
        for (int k = 0; k < L; ++k) {
            const double angle = -2.0 * M_PI * (double)q * (double)k / (double)Ls;
            g_fft_plan.twiddle_r4_re[off + (q - 1) * L + k] = (float)cos(angle);
            g_fft_plan.twiddle_r4_im[off + (q - 1) * L + k] = (float)sin(angle);
        }
    }
}

void fe_fft_plan_init(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;

    int off = 0;
    int L = 1;
    for (int s = 0; s < FE_FFT_R4_STAGES; ++s) {
        g_fft_plan.twiddle_r4_off[s] = off;
        plan_fill_r4_stage(s, L);
        off += 3 * L;
        L *= 4;
    }
    for (int k = 0; k < FE_FFT_HALF / 2; ++k) {
        const double angle = -2.0 * M_PI * (double)k / (double)FE_FFT_HALF;
        g_fft_plan.twiddle_r2_re[k] = (float)cos(angle);
        g_fft_plan.twiddle_r2_im[k] = (float)sin(angle);
    }
    for (int k = 0; k < FE_FFT_HALF; ++k) {
        const double angle = -2.0 * M_PI * (double)k / (double)FE_FFT_N;
        g_fft_plan.realfft_re[k] = (float)cos(angle);
        g_fft_plan.realfft_im[k] = (float)sin(angle);
    }
}

/*
 * Per-tier kernels live in src/fft/fft_<tier>.c, each built with its
 * own -march. This TU resolves the function pointer at runtime.
 */
#include "qgemm/cpu_detect.h"

typedef void (*fe_fft_fwd_fn)(const float *, float *, float *);
typedef void (*fe_fft_inv_fn)(const float *, const float *, float *);

static fe_fft_fwd_fn g_fft_fwd = NULL;
static fe_fft_inv_fn g_fft_inv = NULL;

static void fe_fft_dispatch_select(void) {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64) || \
    defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* arm64: NEON is baseline. */
    g_fft_fwd = fe_fft_forward_neon;
    g_fft_inv = fe_fft_inverse_neon;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* x86 FFT: AVX2 is the floor (matches qgemm tier policy). fe_qgemm_init
     * rejects any host without AVX2 before this runs, so AVX2 is guaranteed
     * present here — no scalar fallback. */
    uint32_t caps = fe_cpu_x86_caps();
    g_fft_fwd = fe_fft_forward_avx2;
    g_fft_inv = fe_fft_inverse_avx2;
    if ((caps & FE_X86_HAS_AVX512VNNI) && (caps & FE_X86_HAS_OS_AVX512)) {
        g_fft_fwd = fe_fft_forward_avx512;
        g_fft_inv = fe_fft_inverse_avx512;
    }
#else
#  error "unsupported ISA — faster-enhancer.c requires aarch64 (NEON) or x86_64 (AVX2)"
#endif
}

void fe_fft_init(void) {
    fe_fft_plan_init();
    if (!g_fft_fwd) fe_fft_dispatch_select();
}

void fe_rfft(const float *in, float *re, float *im) {
    g_fft_fwd(in, re, im);
}

void fe_irfft(const float *re, const float *im, float *out) {
    g_fft_inv(re, im, out);
}

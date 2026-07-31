/*
 * ISA detection + internal API for the 1024-pt real FFT. Hard-coded to
 * N=1024 real / 512 complex, Stockham radix-4 x4 + final radix-2.
 *
 * Per-tier kernels live in fft_<tier>.inl. The shared plan and twiddles
 * are file-static in src/fe_fft.c.
 */
#ifndef FE_FFT_ARCH_H
#define FE_FFT_ARCH_H

#include <stddef.h>

/* Reuse FE_QGEMM_HAVE_* detection. */
#include "../qgemm/qgemm_arch.h"

#define FE_FFT_N         1024            /* real-signal FFT length */
#define FE_FFT_HALF      512             /* N/2: complex FFT length */
#define FE_FFT_R4_STAGES 4               /* radix-4 stages (4^4 = 256) */

/* Shared FFT plan.
 *   twiddle_r4[s][k]: W^k for stage s radix-4 butterflies.
 *   twiddle_r2[k]:    W^k_{HALF} for the final radix-2 stage.
 *   realfft_re/im[k]: W^k_N for real-FFT pre/post processing.
 * Stored as separate re/im arrays for SIMD-friendly loads. */
typedef struct {
    float twiddle_r4_re[256 * 4];
    float twiddle_r4_im[256 * 4];
    int   twiddle_r4_off[FE_FFT_R4_STAGES];

    float twiddle_r2_re[FE_FFT_HALF / 2];
    float twiddle_r2_im[FE_FFT_HALF / 2];

    float realfft_re[FE_FFT_HALF];
    float realfft_im[FE_FFT_HALF];
} fe_fft_plan_t;

/* Plan owned by fe_fft.c. */
extern fe_fft_plan_t g_fft_plan;

void fe_fft_plan_init(void);

/* Per-tier entry points (one selected by the dispatcher). No scalar tier:
 * unsupported ISAs do not run (CMake FATAL_ERRORs non-arm64/x86_64; the x86
 * AVX2+FMA3+F16C runtime floor is enforced at fe_qgemm_init).
 * Declared unconditionally so the dispatcher TU (baseline -march) can
 * take pointers; per-tier .c wrappers (with their own -march) define. */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64) || \
    defined(__ARM_NEON) || defined(__ARM_NEON__)
void fe_fft_forward_neon(const float *in_real, float *out_re, float *out_im);
void fe_fft_inverse_neon(const float *in_re, const float *in_im, float *out_real);
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
/* x86: AVX2 is the baseline (pre-AVX2 tiers dropped to keep cross-tier
 * byte-identity with FMA-using tiers). */
void fe_fft_forward_avx2  (const float *in_real, float *out_re, float *out_im);
void fe_fft_inverse_avx2  (const float *in_re, const float *in_im, float *out_real);
void fe_fft_forward_avx512(const float *in_real, float *out_re, float *out_im);
void fe_fft_inverse_avx512(const float *in_re, const float *in_im, float *out_real);

/* Small-L stage helpers shared between AVX2 and AVX-512 paths.
 * `fft_x86_radix4_stage` (128-bit, L>=4) is exported by fft_avx2.c. */
void fft_x86_radix4_stage      (const float *in_re, const float *in_im,
                                float *out_re, float *out_im,
                                const float *twr, const float *twi, int L);
void fft_avx2_radix4_stage_wide(const float *in_re, const float *in_im,
                                float *out_re, float *out_im,
                                const float *twr, const float *twi, int L);
#endif

#endif /* FE_FFT_ARCH_H */

// fp32 GEMM used by the few non-int8 paths (strided enc_pre, fp32 ConvTranspose).
// NEON / x86 128-bit via fe_simd.h, NR=8 column-blocked pre-packed B.
//
// Design:
//   B is pre-packed once at model load; the inner loop streams linearly.
//   A is read from row-major (lda stride). Activation tensors are small
//   (~200 KB worst case) and already contiguous, so packing A buys nothing.
//   Micro-kernel is a register-resident 8x8 tile with 16 fp32
//   accumulators (NEON has 32 vector regs; plenty of headroom).
//   K loop unrolled 4x for ILP without exploding register pressure.
#include "fe_sgemm.h"
#include "fe_simd.h"
#include <string.h>

// ---- Packing ----
/* Pack W [N, K] (PyTorch linear weight) as if it were W^T [K, N]. */
void fe_pack_W(const float *W, int N, int K, float *packed) {
    const int NR = FE_SGEMM_NR;
    int n_block = 0;
    for (int n = 0; n + NR <= N; n += NR, ++n_block) {
        float *out = packed + (size_t)n_block * (size_t)K * NR;
        for (int k = 0; k < K; ++k) {
            /* W^T[k, n + j] = W[n + j, k]   (stride K on N axis) */
            for (int j = 0; j < NR; ++j) {
                out[(size_t)k * NR + j] = W[(size_t)(n + j) * K + k];
            }
        }
    }
    int n = n_block * NR;
    if (n < N) {
        float *out = packed + (size_t)n_block * (size_t)K * NR;
        for (int k = 0; k < K; ++k) {
            int j = 0;
            for (; j < N - n; ++j) out[(size_t)k * NR + j] = W[(size_t)(n + j) * K + k];
            for (; j < NR; ++j)    out[(size_t)k * NR + j] = 0.0f;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Micro-kernel: 8 rows of A x 8 cols of packed B -> 8x8 tile of C     */
/*  Uses 16 fp32 vector accumulators (32 NEON regs available).         */
/* ------------------------------------------------------------------ */
#if defined(FE_SIMD_NEON)

static inline void fe_sgemm_kernel_8x8(int K,
                                       const float *A, int lda,
                                       const float *Bp,
                                       float beta, float *C, int ldc) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

    /* Outer-product loop: load 4 contiguous K-values for each A-row in one
     * vld1q, then issue 4 K-step FMAs by selecting lanes 0/1/2/3. Replaces
     * the prior 4-scalar gather + lane-insert pattern (which serialized
     * through Firestorm's INS pipe). 8 row-vector loads per 4 K-steps. */
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t aR0 = vld1q_f32(A + 0 * lda + k);
        float32x4_t aR1 = vld1q_f32(A + 1 * lda + k);
        float32x4_t aR2 = vld1q_f32(A + 2 * lda + k);
        float32x4_t aR3 = vld1q_f32(A + 3 * lda + k);
        float32x4_t aR4 = vld1q_f32(A + 4 * lda + k);
        float32x4_t aR5 = vld1q_f32(A + 5 * lda + k);
        float32x4_t aR6 = vld1q_f32(A + 6 * lda + k);
        float32x4_t aR7 = vld1q_f32(A + 7 * lda + k);

        #define KSTEP(KK, LANE)                                              \
        do {                                                                  \
            float32x4_t b0 = vld1q_f32(Bp + (k + KK) * 8);                    \
            float32x4_t b1 = vld1q_f32(Bp + (k + KK) * 8 + 4);                \
            c00 = vfmaq_laneq_f32(c00, b0, aR0, LANE); c01 = vfmaq_laneq_f32(c01, b1, aR0, LANE); \
            c10 = vfmaq_laneq_f32(c10, b0, aR1, LANE); c11 = vfmaq_laneq_f32(c11, b1, aR1, LANE); \
            c20 = vfmaq_laneq_f32(c20, b0, aR2, LANE); c21 = vfmaq_laneq_f32(c21, b1, aR2, LANE); \
            c30 = vfmaq_laneq_f32(c30, b0, aR3, LANE); c31 = vfmaq_laneq_f32(c31, b1, aR3, LANE); \
            c40 = vfmaq_laneq_f32(c40, b0, aR4, LANE); c41 = vfmaq_laneq_f32(c41, b1, aR4, LANE); \
            c50 = vfmaq_laneq_f32(c50, b0, aR5, LANE); c51 = vfmaq_laneq_f32(c51, b1, aR5, LANE); \
            c60 = vfmaq_laneq_f32(c60, b0, aR6, LANE); c61 = vfmaq_laneq_f32(c61, b1, aR6, LANE); \
            c70 = vfmaq_laneq_f32(c70, b0, aR7, LANE); c71 = vfmaq_laneq_f32(c71, b1, aR7, LANE); \
        } while (0)
        KSTEP(0, 0);
        KSTEP(1, 1);
        KSTEP(2, 2);
        KSTEP(3, 3);
        #undef KSTEP
    }
    for (; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(Bp + k * 8);
        float32x4_t b1 = vld1q_f32(Bp + k * 8 + 4);
        float a0v = A[0 * lda + k], a1v = A[1 * lda + k],
              a2v = A[2 * lda + k], a3v = A[3 * lda + k],
              a4v = A[4 * lda + k], a5v = A[5 * lda + k],
              a6v = A[6 * lda + k], a7v = A[7 * lda + k];
        c00 = vfmaq_n_f32(c00, b0, a0v); c01 = vfmaq_n_f32(c01, b1, a0v);
        c10 = vfmaq_n_f32(c10, b0, a1v); c11 = vfmaq_n_f32(c11, b1, a1v);
        c20 = vfmaq_n_f32(c20, b0, a2v); c21 = vfmaq_n_f32(c21, b1, a2v);
        c30 = vfmaq_n_f32(c30, b0, a3v); c31 = vfmaq_n_f32(c31, b1, a3v);
        c40 = vfmaq_n_f32(c40, b0, a4v); c41 = vfmaq_n_f32(c41, b1, a4v);
        c50 = vfmaq_n_f32(c50, b0, a5v); c51 = vfmaq_n_f32(c51, b1, a5v);
        c60 = vfmaq_n_f32(c60, b0, a6v); c61 = vfmaq_n_f32(c61, b1, a6v);
        c70 = vfmaq_n_f32(c70, b0, a7v); c71 = vfmaq_n_f32(c71, b1, a7v);
    }

    if (beta == 0.0f) {
        vst1q_f32(C + 0 * ldc + 0, c00); vst1q_f32(C + 0 * ldc + 4, c01);
        vst1q_f32(C + 1 * ldc + 0, c10); vst1q_f32(C + 1 * ldc + 4, c11);
        vst1q_f32(C + 2 * ldc + 0, c20); vst1q_f32(C + 2 * ldc + 4, c21);
        vst1q_f32(C + 3 * ldc + 0, c30); vst1q_f32(C + 3 * ldc + 4, c31);
        vst1q_f32(C + 4 * ldc + 0, c40); vst1q_f32(C + 4 * ldc + 4, c41);
        vst1q_f32(C + 5 * ldc + 0, c50); vst1q_f32(C + 5 * ldc + 4, c51);
        vst1q_f32(C + 6 * ldc + 0, c60); vst1q_f32(C + 6 * ldc + 4, c61);
        vst1q_f32(C + 7 * ldc + 0, c70); vst1q_f32(C + 7 * ldc + 4, c71);
    } else {
        vst1q_f32(C + 0 * ldc + 0, vaddq_f32(vld1q_f32(C + 0 * ldc + 0), c00));
        vst1q_f32(C + 0 * ldc + 4, vaddq_f32(vld1q_f32(C + 0 * ldc + 4), c01));
        vst1q_f32(C + 1 * ldc + 0, vaddq_f32(vld1q_f32(C + 1 * ldc + 0), c10));
        vst1q_f32(C + 1 * ldc + 4, vaddq_f32(vld1q_f32(C + 1 * ldc + 4), c11));
        vst1q_f32(C + 2 * ldc + 0, vaddq_f32(vld1q_f32(C + 2 * ldc + 0), c20));
        vst1q_f32(C + 2 * ldc + 4, vaddq_f32(vld1q_f32(C + 2 * ldc + 4), c21));
        vst1q_f32(C + 3 * ldc + 0, vaddq_f32(vld1q_f32(C + 3 * ldc + 0), c30));
        vst1q_f32(C + 3 * ldc + 4, vaddq_f32(vld1q_f32(C + 3 * ldc + 4), c31));
        vst1q_f32(C + 4 * ldc + 0, vaddq_f32(vld1q_f32(C + 4 * ldc + 0), c40));
        vst1q_f32(C + 4 * ldc + 4, vaddq_f32(vld1q_f32(C + 4 * ldc + 4), c41));
        vst1q_f32(C + 5 * ldc + 0, vaddq_f32(vld1q_f32(C + 5 * ldc + 0), c50));
        vst1q_f32(C + 5 * ldc + 4, vaddq_f32(vld1q_f32(C + 5 * ldc + 4), c51));
        vst1q_f32(C + 6 * ldc + 0, vaddq_f32(vld1q_f32(C + 6 * ldc + 0), c60));
        vst1q_f32(C + 6 * ldc + 4, vaddq_f32(vld1q_f32(C + 6 * ldc + 4), c61));
        vst1q_f32(C + 7 * ldc + 0, vaddq_f32(vld1q_f32(C + 7 * ldc + 0), c70));
        vst1q_f32(C + 7 * ldc + 4, vaddq_f32(vld1q_f32(C + 7 * ldc + 4), c71));
    }
}

/* Bias-seeded variant: identical FMA loop, but accumulators are
 * initialized from bias[nr..nr+7] (broadcast across all 8 rows) instead
 * of zero. Eliminates the bias-broadcast pre-pass and the beta=1 RMW
 * store path used by every Conv/Linear call. */
static inline void fe_sgemm_kernel_8x8_bias(int K,
                                            const float *A, int lda,
                                            const float *Bp,
                                            const float *bias,
                                            float *C, int ldc) {
    float32x4_t b0 = vld1q_f32(bias + 0);
    float32x4_t b1 = vld1q_f32(bias + 4);
    float32x4_t c00 = b0, c01 = b1;
    float32x4_t c10 = b0, c11 = b1;
    float32x4_t c20 = b0, c21 = b1;
    float32x4_t c30 = b0, c31 = b1;
    float32x4_t c40 = b0, c41 = b1;
    float32x4_t c50 = b0, c51 = b1;
    float32x4_t c60 = b0, c61 = b1;
    float32x4_t c70 = b0, c71 = b1;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        float32x4_t aR0 = vld1q_f32(A + 0 * lda + k);
        float32x4_t aR1 = vld1q_f32(A + 1 * lda + k);
        float32x4_t aR2 = vld1q_f32(A + 2 * lda + k);
        float32x4_t aR3 = vld1q_f32(A + 3 * lda + k);
        float32x4_t aR4 = vld1q_f32(A + 4 * lda + k);
        float32x4_t aR5 = vld1q_f32(A + 5 * lda + k);
        float32x4_t aR6 = vld1q_f32(A + 6 * lda + k);
        float32x4_t aR7 = vld1q_f32(A + 7 * lda + k);
        #define KSTEP_B(KK, LANE)                                            \
        do {                                                                  \
            float32x4_t bb0 = vld1q_f32(Bp + (k + KK) * 8);                   \
            float32x4_t bb1 = vld1q_f32(Bp + (k + KK) * 8 + 4);               \
            c00 = vfmaq_laneq_f32(c00, bb0, aR0, LANE); c01 = vfmaq_laneq_f32(c01, bb1, aR0, LANE); \
            c10 = vfmaq_laneq_f32(c10, bb0, aR1, LANE); c11 = vfmaq_laneq_f32(c11, bb1, aR1, LANE); \
            c20 = vfmaq_laneq_f32(c20, bb0, aR2, LANE); c21 = vfmaq_laneq_f32(c21, bb1, aR2, LANE); \
            c30 = vfmaq_laneq_f32(c30, bb0, aR3, LANE); c31 = vfmaq_laneq_f32(c31, bb1, aR3, LANE); \
            c40 = vfmaq_laneq_f32(c40, bb0, aR4, LANE); c41 = vfmaq_laneq_f32(c41, bb1, aR4, LANE); \
            c50 = vfmaq_laneq_f32(c50, bb0, aR5, LANE); c51 = vfmaq_laneq_f32(c51, bb1, aR5, LANE); \
            c60 = vfmaq_laneq_f32(c60, bb0, aR6, LANE); c61 = vfmaq_laneq_f32(c61, bb1, aR6, LANE); \
            c70 = vfmaq_laneq_f32(c70, bb0, aR7, LANE); c71 = vfmaq_laneq_f32(c71, bb1, aR7, LANE); \
        } while (0)
        KSTEP_B(0, 0); KSTEP_B(1, 1); KSTEP_B(2, 2); KSTEP_B(3, 3);
        #undef KSTEP_B
    }
    for (; k < K; ++k) {
        float32x4_t bb0 = vld1q_f32(Bp + k * 8);
        float32x4_t bb1 = vld1q_f32(Bp + k * 8 + 4);
        float a0v = A[0 * lda + k], a1v = A[1 * lda + k],
              a2v = A[2 * lda + k], a3v = A[3 * lda + k],
              a4v = A[4 * lda + k], a5v = A[5 * lda + k],
              a6v = A[6 * lda + k], a7v = A[7 * lda + k];
        c00 = vfmaq_n_f32(c00, bb0, a0v); c01 = vfmaq_n_f32(c01, bb1, a0v);
        c10 = vfmaq_n_f32(c10, bb0, a1v); c11 = vfmaq_n_f32(c11, bb1, a1v);
        c20 = vfmaq_n_f32(c20, bb0, a2v); c21 = vfmaq_n_f32(c21, bb1, a2v);
        c30 = vfmaq_n_f32(c30, bb0, a3v); c31 = vfmaq_n_f32(c31, bb1, a3v);
        c40 = vfmaq_n_f32(c40, bb0, a4v); c41 = vfmaq_n_f32(c41, bb1, a4v);
        c50 = vfmaq_n_f32(c50, bb0, a5v); c51 = vfmaq_n_f32(c51, bb1, a5v);
        c60 = vfmaq_n_f32(c60, bb0, a6v); c61 = vfmaq_n_f32(c61, bb1, a6v);
        c70 = vfmaq_n_f32(c70, bb0, a7v); c71 = vfmaq_n_f32(c71, bb1, a7v);
    }
    vst1q_f32(C + 0 * ldc + 0, c00); vst1q_f32(C + 0 * ldc + 4, c01);
    vst1q_f32(C + 1 * ldc + 0, c10); vst1q_f32(C + 1 * ldc + 4, c11);
    vst1q_f32(C + 2 * ldc + 0, c20); vst1q_f32(C + 2 * ldc + 4, c21);
    vst1q_f32(C + 3 * ldc + 0, c30); vst1q_f32(C + 3 * ldc + 4, c31);
    vst1q_f32(C + 4 * ldc + 0, c40); vst1q_f32(C + 4 * ldc + 4, c41);
    vst1q_f32(C + 5 * ldc + 0, c50); vst1q_f32(C + 5 * ldc + 4, c51);
    vst1q_f32(C + 6 * ldc + 0, c60); vst1q_f32(C + 6 * ldc + 4, c61);
    vst1q_f32(C + 7 * ldc + 0, c70); vst1q_f32(C + 7 * ldc + 4, c71);
}

#endif /* FE_SIMD_NEON */

/* ------------------------------------------------------------------ */
/*  x86 AVX2 8x8 micro-kernel (FMA3, ymm accumulators).                 */
/* ------------------------------------------------------------------ */
#if (defined(__AVX2__) || defined(__AVXVNNI__) || defined(__AVX512VNNI__)) && \
    !defined(FE_SIMD_NEON)
#include <immintrin.h>

static inline void fe_sgemm_kernel_8x8_avx2(int K,
                                            const float *A, int lda,
                                            const float *Bp,
                                            float beta, float *C, int ldc) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 b = _mm256_loadu_ps(Bp + (size_t)k * 8);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 0 * lda + k), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 1 * lda + k), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * lda + k), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * lda + k), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4 * lda + k), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5 * lda + k), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 6 * lda + k), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 7 * lda + k), b, c7);
    }
    if (beta != 0.0f) {
        c0 = _mm256_add_ps(c0, _mm256_loadu_ps(C + 0 * ldc));
        c1 = _mm256_add_ps(c1, _mm256_loadu_ps(C + 1 * ldc));
        c2 = _mm256_add_ps(c2, _mm256_loadu_ps(C + 2 * ldc));
        c3 = _mm256_add_ps(c3, _mm256_loadu_ps(C + 3 * ldc));
        c4 = _mm256_add_ps(c4, _mm256_loadu_ps(C + 4 * ldc));
        c5 = _mm256_add_ps(c5, _mm256_loadu_ps(C + 5 * ldc));
        c6 = _mm256_add_ps(c6, _mm256_loadu_ps(C + 6 * ldc));
        c7 = _mm256_add_ps(c7, _mm256_loadu_ps(C + 7 * ldc));
    }
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}

static inline void fe_sgemm_kernel_8x8_bias_avx2(int K,
                                                 const float *A, int lda,
                                                 const float *Bp,
                                                 const float *bias,
                                                 float *C, int ldc) {
    __m256 b = _mm256_loadu_ps(bias);
    __m256 c0 = b, c1 = b, c2 = b, c3 = b, c4 = b, c5 = b, c6 = b, c7 = b;
    for (int k = 0; k < K; ++k) {
        __m256 bv = _mm256_loadu_ps(Bp + (size_t)k * 8);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 0 * lda + k), bv, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 1 * lda + k), bv, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * lda + k), bv, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * lda + k), bv, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4 * lda + k), bv, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5 * lda + k), bv, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 6 * lda + k), bv, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 7 * lda + k), bv, c7);
    }
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}

#define FE_SGEMM_HAVE_AVX2 1
#endif

/* ------------------------------------------------------------------ */
/*  Generic 8x8 reference (x86 baseline fp32 kernel + universal tail)   */
/* ------------------------------------------------------------------ */
static void fe_sgemm_kernel_8x8_ref(int M, int N, int K,
                                    const float *A, int lda,
                                    const float *Bp,
                                    float beta, float *C, int ldc) {
    float acc[8][8] = {{0}};
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            float av = A[i * lda + k];
            for (int j = 0; j < N; ++j) acc[i][j] += av * Bp[k * 8 + j];
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float cur = (beta == 0.0f) ? 0.0f : C[i * ldc + j];
            C[i * ldc + j] = cur + acc[i][j];
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Driver                                                             */
/* ------------------------------------------------------------------ */
void fe_sgemm_packed(int M, int N, int K,
                     const float *A, int lda,
                     const float *Bp,
                     float beta, float *C, int ldc) {
    const int MR = FE_SGEMM_MR;
    const int NR = FE_SGEMM_NR;

    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
#if defined(FE_SIMD_NEON)
            fe_sgemm_kernel_8x8(K,
                                A + (size_t)mr * lda, lda,
                                Bp + (size_t)nb * (size_t)K * NR,
                                beta,
                                C + (size_t)mr * ldc + nr, ldc);
#elif defined(FE_SGEMM_HAVE_AVX2)
            fe_sgemm_kernel_8x8_avx2(K,
                                     A + (size_t)mr * lda, lda,
                                     Bp + (size_t)nb * (size_t)K * NR,
                                     beta,
                                     C + (size_t)mr * ldc + nr, ldc);
#else
            fe_sgemm_kernel_8x8_ref(MR, NR, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    beta,
                                    C + (size_t)mr * ldc + nr, ldc);
#endif
        }
        /* N tail (N not divisible by NR). */
        if (nr < N) {
            fe_sgemm_kernel_8x8_ref(MR, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    beta,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
    /* M tail (M not divisible by MR). Not expected for current model shapes. */
    if (mr < M) {
        int M_tail = M - mr;
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
            fe_sgemm_kernel_8x8_ref(M_tail, NR, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    beta,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) {
            fe_sgemm_kernel_8x8_ref(M_tail, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    beta,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
}

void fe_sgemm_packed_bias(int M, int N, int K,
                          const float *A, int lda,
                          const float *Bp,
                          const float *bias,
                          float *C, int ldc) {
    const int MR = FE_SGEMM_MR;
    const int NR = FE_SGEMM_NR;
#if defined(FE_SIMD_NEON)
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
            fe_sgemm_kernel_8x8_bias(K,
                                     A + (size_t)mr * lda, lda,
                                     Bp + (size_t)nb * (size_t)K * NR,
                                     bias + nr,
                                     C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) {
            /* Tail: bias-broadcast then beta=1 accumulate. */
            for (int i = 0; i < MR; ++i)
                for (int j = nr; j < N; ++j) C[(size_t)(mr + i) * ldc + j] = bias[j];
            fe_sgemm_kernel_8x8_ref(MR, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
    if (mr < M) {
        int M_tail = M - mr;
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
            for (int i = 0; i < M_tail; ++i)
                for (int j = 0; j < NR; ++j) C[(size_t)(mr + i) * ldc + nr + j] = bias[nr + j];
            fe_sgemm_kernel_8x8_ref(M_tail, NR, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) {
            for (int i = 0; i < M_tail; ++i)
                for (int j = nr; j < N; ++j) C[(size_t)(mr + i) * ldc + j] = bias[j];
            fe_sgemm_kernel_8x8_ref(M_tail, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
#elif defined(FE_SGEMM_HAVE_AVX2)
    int mr = 0;
    for (; mr + MR <= M; mr += MR) {
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
            fe_sgemm_kernel_8x8_bias_avx2(K,
                                          A + (size_t)mr * lda, lda,
                                          Bp + (size_t)nb * (size_t)K * NR,
                                          bias + nr,
                                          C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) {
            for (int i = 0; i < MR; ++i)
                for (int j = nr; j < N; ++j) C[(size_t)(mr + i) * ldc + j] = bias[j];
            fe_sgemm_kernel_8x8_ref(MR, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
    if (mr < M) {
        int M_tail = M - mr;
        int nr = 0, nb = 0;
        for (; nr + NR <= N; nr += NR, ++nb) {
            for (int i = 0; i < M_tail; ++i)
                for (int j = 0; j < NR; ++j) C[(size_t)(mr + i) * ldc + nr + j] = bias[nr + j];
            fe_sgemm_kernel_8x8_ref(M_tail, NR, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
        if (nr < N) {
            for (int i = 0; i < M_tail; ++i)
                for (int j = nr; j < N; ++j) C[(size_t)(mr + i) * ldc + j] = bias[j];
            fe_sgemm_kernel_8x8_ref(M_tail, N - nr, K,
                                    A + (size_t)mr * lda, lda,
                                    Bp + (size_t)nb * (size_t)K * NR,
                                    1.0f,
                                    C + (size_t)mr * ldc + nr, ldc);
        }
    }
#else
    /* Scalar fallback: plain GEMM then bias add. */
    fe_sgemm_packed(M, N, K, A, lda, Bp, 0.0f, C, ldc);
    for (int m = 0; m < M; ++m) {
        float *row = C + (size_t)m * ldc;
        for (int n = 0; n < N; ++n) row[n] += bias[n];
    }
#endif
}

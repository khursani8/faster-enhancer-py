// Hand-written fp32 GEMM for the few non-int8 paths. Pre-packed B
// (NR=8-blocked, packed once at load), 8x8 register-resident micro-kernel.
// All hot sizes in this model have M/N divisible by 8; the micro-kernel
// handles any K remainder internally.
#ifndef FE_SGEMM_H
#define FE_SGEMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FE_SGEMM_NR 8     /* N-tile width (must match micro-kernel) */
#define FE_SGEMM_MR 8     /* M-tile height                          */

/* Bytes of packed storage required for a [K x N] B matrix.
 * N is padded up to a multiple of NR.                              */
static inline size_t fe_sgemm_packed_size(int K, int N) {
    int Np = (N + FE_SGEMM_NR - 1) & ~(FE_SGEMM_NR - 1);
    return (size_t)K * (size_t)Np * sizeof(float);
}

/* Pack a [N x K] row-major matrix (PyTorch [out, in] linear weight)
 * into the same form: transposes on the fly so the result acts as
 * B = W^T when used in fe_sgemm.                                   */
void fe_pack_W(const float *W, int N, int K, float *packed);

/* C[M, N] = beta * C + A[M, K] @ B_packed.
 *   A is row-major [M, K] with leading dim = lda (K typical).
 *   C is row-major [M, N] with leading dim = ldc (N typical).
 *   beta must be 0.0f or 1.0f.                                     */
void fe_sgemm_packed(int M, int N, int K,
                     const float *A, int lda,
                     const float *Bp,
                     float beta, float *C, int ldc);

/* Same as fe_sgemm_packed but seeds the accumulators with `bias[n]` per
 * column instead of zero. beta is implicitly 0 (writes Y, no RMW).      *
 * `bias` must point to N floats (the column-axis bias broadcast).      */
void fe_sgemm_packed_bias(int M, int N, int K,
                          const float *A, int lda,
                          const float *Bp,
                          const float *bias,
                          float *C, int ldc);

/* PyTorch nn.Linear with pre-packed W^T (B in fe_sgemm_packed).
 *   Y[M, N] = X[M, K] @ Wp  (+ bias[N])                            */
static inline void fe_linear_packed(int M, int N, int K,
                                    const float *X, const float *Wp,
                                    const float *bias, float *Y) {
    if (bias) {
        fe_sgemm_packed_bias(M, N, K, X, K, Wp, bias, Y, N);
    } else {
        fe_sgemm_packed(M, N, K, X, K, Wp, 0.0f, Y, N);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* FE_SGEMM_H */

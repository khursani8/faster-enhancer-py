/*
 * Public int8 W8A8 GEMM API. Per-channel symmetric int8 weights
 * (int8 + fp32 scale[N]) x per-tensor asymmetric uint8 activations
 * stored as int8 = u8 - 128. int32 accumulate -> fp32 dequant.
 * K must be a multiple of 4 (DOTPROD). I8MM kernels require K%8==0; the
 * K=20 attention score path is deliberately routed through DOTPROD even on
 * I8MM-capable CPUs. Runtime SIMD dispatch is in qgemm_dispatch.c.
 */
#ifndef FE_QGEMM_H
#define FE_QGEMM_H

#include <stdint.h>
#include <stddef.h>
#include "fe_calib.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FE_QGEMM_MR 8
#define FE_QGEMM_NR 8

/* Packed int8 weight layout (NR-blocked along N for DOTPROD streaming):
 *   packed[block_n*K*NR + k*NR + n_lane] = Q[k, block_n*NR + n_lane]
 * scales[N] is the per-channel fp32 scale in the same N order. */

/* Compute packed buffer size (bytes) for an [N, K] int8 weight matrix. */
static inline size_t fe_qgemm_packed_size(int K, int N) {
    int Np = (N + FE_QGEMM_NR - 1) & ~(FE_QGEMM_NR - 1);
    return (size_t)K * (size_t)Np * sizeof(int8_t);
}

/* Quantize a [N, K] fp32 weight (PyTorch [out, in]) per-channel
 * symmetric int8 along axis 0; outputs packed int8 + scales[N]. */
void fe_qgemm_pack_W(const float *W_fp32, int N, int K,
                     int8_t *packed_out, float *scales_out);

/* Per-channel sum of packed int8 weights. Required for the asymmetric
 * uint8 (128 - zp_x) * row_sums[n] dequant correction. Layout-invariant
 * across DOTPROD/I8MM packs. */
void fe_qgemm_compute_row_sums(const int8_t *packed, int N, int K,
                               int32_t *row_sums_out);

/* SIMD bias_eff builder for the bias=NULL GRU path: writes
 *   bias_eff_buf[n] = combined_scale[n] * k * (float)row_sums[n]
 * where k = (float)(128 - zp_x). Matches scalar order; byte-id with
 * the scalar `combined * k * row_sums` evaluation. */
void fe_qgemm_build_bias_eff_nobias(const float *combined_scale,
                                    const int32_t *row_sums,
                                    float k, int N,
                                    float *bias_eff_buf);

/* DOTPROD -> I8MM in-place repack of all full N-blocks. No-op on
 * non-aarch64 or when K is not a multiple of 8. */
void fe_qgemm_repack_i8mm(int8_t *packed, int N, int K);

/* Single-block DOTPROD -> I8MM repack (used by on-the-fly K/V packs). */
void fe_qgemm_i8mm_repack_block(const int8_t *src, int K, int8_t *dst);

/* A already quantized (scale_x, zp_x). Dequant:
 *   y[m,n] = scale_x*scales_w[n]*(C32[m,n] + (128-zp_x)*row_sums[n])
 *          + bias[n]
 * NULL row_sums + zp_x=128 -> symmetric fallback. */
void fe_qgemm_prequant(int M, int N, int K,
                       const int8_t *A_q, float scale_x, int32_t zp_x,
                       const int8_t *Bp,
                       const float  *scales_w,
                       const int32_t *row_sums,
                       const float  *bias,
                       float *C, int ldc,
                       int32_t *c32_scratch,
                       int act_silu);

/* Quantize fp32 [M*K] -> int8 with asymmetric uint8 semantics
 * (scale = (max-min)/255, zp = clamp(round(-min/scale))). Stored as
 * int8 = u8 - 128. Returns (scale, zp). */
FeActQuant fe_quantize_activation(const float *A, int total, int8_t *A_q);

/* fp16-input variant — scans + quantizes directly from fp16
 * storage, no fp32 intermediate. Same semantics as fe_quantize_activation. */
FeActQuant fe_quantize_activation_fp16(const uint16_t *A_fp16, int total,
                                        int8_t *A_q);

/* Quantize with caller-provided (scale, zp); skips min/max scan. */
void fe_quantize_activation_with_scale(const float *A, int total,
                                       int8_t *A_q, float scale,
                                       int32_t zp);

/* Transpose-quantize: fp32 [K, M] -> int8 [M, K]. Same contract as
 * fe_quantize_activation but folds the transpose into the quantize. */
FeActQuant fe_quantize_activation_transposed(const float *A_KM,
                                              int M, int K,
                                              int8_t *A_q_MK);

/* fp16-input transpose-quantize. Reads fp16 [K, M],
 * writes int8 [M, K]. On-the-fly fp16->fp32 cvt; no fp32 scratch. */
FeActQuant fe_quantize_activation_transposed_fp16(const uint16_t *A_KM_fp16,
                                                   int M, int K,
                                                   int8_t *A_q_MK);

/* Calibration-tracking wrappers. Activation quant is dynamic per frame;
 * FeActScale tracks running min/max for diagnostics only. Input is [K, M];
 * transpose folds into quantize. */
void fe_qgemm_packed_calib_transposed_in(int M, int N, int K,
                                          const float *A_KM,
                                          const int8_t *Bp,
                                          const float  *scales_w,
                                          const int32_t *row_sums,
                                          const float  *bias,
                                          float *C, int ldc,
                                          int8_t  *aq_scratch,
                                          int32_t *c32_scratch,
                                          FeActScale *act);

/* fp16-input transposed_in. A_KM is uint16_t* (fp16). */
void fe_qgemm_packed_calib_transposed_in_fp16(int M, int N, int K,
                                               const uint16_t *A_KM_fp16,
                                               const int8_t *Bp,
                                               const float  *scales_w,
                                               const int32_t *row_sums,
                                               const float  *bias,
                                               float *C, int ldc,
                                               int8_t  *aq_scratch,
                                               int32_t *c32_scratch,
                                               FeActScale *act);

/* A fp32, B fp16. Joint min/max + quantize w/o fp32
 * intermediate for B. Eliminates the engine-side dec_1x1 unpack pass. */
void fe_qgemm_packed_silu_calib_concat2_fp16b(int M, int N, int K_half,
                                               const float *A,
                                               const uint16_t *B_fp16,
                                               const int8_t *Bp,
                                               const float  *scales_w,
                                               const int32_t *row_sums,
                                               const float  *bias,
                                               float *C, int ldc,
                                               int8_t  *aq_scratch,
                                               int32_t *c32_scratch,
                                               FeActScale *act);

/* Accumulating variant: C[m,n] += bias[n] + dequant(C32[m,n], ...).
 * Used to fuse RNNFormer residual adds into the FC epilogue. */
void fe_qgemm_packed_calib_acc(int M, int N, int K,
                                const float *A,
                                const int8_t *Bp,
                                const float  *scales_w,
                                const int32_t *row_sums,
                                const float  *bias,
                                float *C, int ldc,
                                int8_t  *aq_scratch,
                                int32_t *c32_scratch,
                                float   *fp32_scratch,
                                FeActScale *act);

/* fp32 -> int8 wrapper used by MHSA QKV. Asymmetric uint8 input,
 * symmetric int8 output (zp=128) so downstream Q*K^T / S*V see
 * zero-centered operands. Returns the output (scale, zp=128). */
FeActQuant fe_qgemm_packed_calib_to_int8out(int M, int N, int K,
                                       const float *A,
                                       const int8_t *Bp,
                                       const float  *scales_w,
                                       const int32_t *row_sums,
                                       const float  *bias,
                                       int8_t *C_q, int ldc,
                                       float  *C_fp32_scratch, int ldc_fp32,
                                       int8_t  *aq_scratch,
                                       int32_t *c32_scratch,
                                       FeActScale *act_in,
                                       FeActScale *act_out);

#ifdef __cplusplus
}
#endif

#endif /* FE_QGEMM_H */

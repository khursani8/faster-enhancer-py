/*
 * Per-tier qgemm kernel prototypes.
 *
 * Each tier has its own .c file built with the matching -march flag (set
 * via CMake set_source_files_properties); qgemm_dispatch.c installs the
 * right kernel pointers at fe_init.
 *
 * Naming:
 *   qgemm_<isa>_int32      - int8 x int8 -> int32 accumulator
 *   qgemm_<isa>_fp32_fused - dequant + bias + SiLU + fp32 store
 *
 * Prototypes are gated only by the host arch family, not by feature
 * macros like __ARM_FEATURE_MATMUL_INT8, so the dispatcher (compiled with
 * baseline flags) can take a kernel's address.
 */
#ifndef FE_ARCH_KERNELS_H
#define FE_ARCH_KERNELS_H

#include <stdint.h>
#include "qgemm_arch.h"
#include "../internal/fe_qgemm.h"

/* Shared GRU r/z gate scratch; only one tier runs at a time. */
extern float fe_gru_r_band[FE_QGEMM_GRU_MR * FE_QGEMM_MAX_GRU_D];
extern float fe_gru_z_band[FE_QGEMM_GRU_MR * FE_QGEMM_MAX_GRU_D];

#ifdef __cplusplus
extern "C" {
#endif

/* ---- SIMD-remainder tail guard ----
 * The scalar tail kernel was removed: every production GEMM has M and N that
 * are exact multiples of the FE_QGEMM_MR/NR tile (enforced at compile time by
 * the _Static_assert block in fe_config_medium.h), so the M%MR / N%NR remainder
 * paths are unreachable. If a future (mis)configuration ever produces a non-tile
 * GEMM, the kernels call this instead of silently dropping the remainder. */
void fe_qgemm_tail_unsupported(void);

/* ARM tiers. */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

void qgemm_neon_int32      (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
/* K=20 specialisation for MHSA Q@K^T. */
void qgemm_neon_int32_k20  (int M, int N,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_neon_fp32_fused (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc, int act_silu, int32_t *c32_tail);
void qgemm_neon_fp32_fused_acc(int M, int N, int K,
                                const int8_t *A_q, const int8_t *Bp,
                                const float *combined_scale, const float *bias,
                                float *C, int ldc, int32_t *c32_tail);
void qgemm_neon_fp32_fused_track_maxabs(int M, int N, int K,
                                         const int8_t *A_q, const int8_t *Bp,
                                         const float *combined_scale,
                                         const float *bias,
                                         float *C, int ldc,
                                         int32_t *c32_tail,
                                         float *max_abs_out);
void qgemm_dotprod_int32      (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
/* K-fixed specialisation for MHSA Q@K^T (K = HD_PAD_K, 20 on Medium).
 * The 5-iter loop fully unrolls and gives the scheduler a clean DOTPROD
 * dependency chain. */
void qgemm_dotprod_int32_k20  (int M, int N,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_dotprod_fp32_fused (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc, int act_silu, int32_t *c32_tail);
/* DOTPROD-tier fused fp32-out variants for DOTPROD-only CPUs. */
void qgemm_dotprod_fp32_fused_acc(int M, int N, int K,
                                const int8_t *A_q, const int8_t *Bp,
                                const float *combined_scale, const float *bias,
                                float *C, int ldc, int32_t *c32_tail);
void qgemm_dotprod_fp32_fused_track_maxabs(int M, int N, int K,
                                         const int8_t *A_q, const int8_t *Bp,
                                         const float *combined_scale,
                                         const float *bias,
                                         float *C, int ldc,
                                         int32_t *c32_tail,
                                         float *max_abs_out);
/* Pre-fault the DOTPROD row-quad BSS scratch pages. Called once
 * from fe_qgemm_init on aarch64, mirror of qgemm_arm_i8mm_prefault_buffers. */
void qgemm_arm_dotprod_prefault_buffers(void);
/* Pre-fault I8MM row-pair BSS scratch pages. Called once from
 * fe_qgemm_init on aarch64, mirror of qgemm_avx2_prefault_buffers. */
void qgemm_arm_i8mm_prefault_buffers(void);
void qgemm_i8mm_int32     (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_i8mm_fp32_fused(int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc, int act_silu, int32_t *c32_tail);

/* Accumulating epilogue: C[m,n] += bias[n] + combined_scale[n]*c32[m,n].
 * Used by rnn_fc / attn_fc to fold the residual add into the dequant pass.
 * Op order matches the unfused path (vfmaq then vaddq). */
void qgemm_i8mm_fp32_fused_acc(int M, int N, int K,
                                 const int8_t *A_q, const int8_t *Bp,
                                 const float *combined_scale,
                                 const float *bias,
                                 float *C, int ldc,
                                 int32_t *c32_tail);

/* Like fp32_fused but also writes max(|C[m,n]|) over the tile into
 * *max_abs_out. Used by the MHSA QKV projection to fold the max-abs
 * scan into the GEMM epilogue. No SiLU: QKV is a pure linear. */
void qgemm_i8mm_fp32_fused_track_maxabs(int M, int N, int K,
                                          const int8_t *A_q, const int8_t *Bp,
                                          const float *combined_scale,
                                          const float *bias,
                                          float *C, int ldc,
                                          int32_t *c32_tail,
                                          float *max_abs_out);

/*
 * GRU full fusion: W_ih x, W_hh h and the gate update in one pass.
 *
 *   Xq, Hq        - caller-quantized int8 inputs
 *   Wq_ih, Wq_hh  - I8MM-packed [3D, D] weights
 *   combined_*    - scale_act * scales_w[n]
 *   bias_eff_*    - combined[n] * (128 - zp_act) * row_sums[n]
 *   br/bz/bn_*    - GRU bias slices, [D] each
 *   h_inout       - h_prev in, h_new out (in place)
 */
/* ARM fp16 inout variants: gru_h stored fp16, dual-store ngate epilogue
 * writes state back as fp16. Same contract as the x86 ones — fp16 storage
 * read/write + fp32 scratch out. */
void qgemm_i8mm_gru_full_fused_fp16inout(int M, int D,
                                          const int8_t *Xq, int ld_x,
                                          const int8_t *Hq, int ld_h,
                                          const int8_t *Wq_ih,
                                          const int8_t *Wq_hh,
                                          const float *combined_ih,
                                          const float *bias_eff_ih,
                                          const float *combined_hh,
                                          const float *bias_eff_hh,
                                          const float *br_sum, const float *bz_sum,
                                          const float *bn_i,   const float *bn_h,
                                          uint16_t *h_inout_fp16, int ld_h_inout,
                                          float *h_out_scratch, int ld_h_out);
void qgemm_dotprod_gru_full_fused_fp16inout(int M, int D,
                                             const int8_t *Xq, int ld_x,
                                             const int8_t *Hq, int ld_h,
                                             const int8_t *Wq_ih,
                                             const int8_t *Wq_hh,
                                             const float *combined_ih,
                                             const float *bias_eff_ih,
                                             const float *combined_hh,
                                             const float *bias_eff_hh,
                                             const float *br_sum, const float *bz_sum,
                                             const float *bn_i,   const float *bn_h,
                                             uint16_t *h_inout_fp16, int ld_h_inout,
                                             float *h_out_scratch, int ld_h_out);
void qgemm_neon_gru_full_fused_fp16inout(int M, int D,
                                          const int8_t *Xq, int ld_x,
                                          const int8_t *Hq, int ld_h,
                                          const int8_t *Wq_ih,
                                          const int8_t *Wq_hh,
                                          const float *combined_ih,
                                          const float *bias_eff_ih,
                                          const float *combined_hh,
                                          const float *bias_eff_hh,
                                          const float *br_sum, const float *bz_sum,
                                          const float *bn_i,   const float *bn_h,
                                          uint16_t *h_inout_fp16, int ld_h_inout,
                                          float *h_out_scratch, int ld_h_out);

#endif /* aarch64 */

/* x86 tiers. AVX2 is the minimum; pre-AVX2 x86 tiers were dropped (no FMA3
 * → breaks cross-tier byte-identity with AVX2+). */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

/* Pre-fault every page of the AVX2 BSS scratch buffers to
 * eliminate first-touch page-fault spikes from steady-state percentiles.
 * No-op on non-AVX2 tiers. Called once from fe_qgemm_init. */
void qgemm_avx2_prefault_buffers(void);

void qgemm_avx2_int32      (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_avx2_int32_k20  (int M, int N,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_avx2_fp32_fused (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            const float *combined_scale, const float *bias,
                            float *C, int ldc, int act_silu, int32_t *c32_tail);
void qgemm_avx2_fp32_fused_acc(int M, int N, int K,
                               const int8_t *A_q, const int8_t *Bp,
                               const float *combined_scale, const float *bias,
                               float *C, int ldc, int32_t *c32_tail);
void qgemm_avx2_fp32_fused_track_maxabs(int M, int N, int K,
                                        const int8_t *A_q, const int8_t *Bp,
                                        const float *combined_scale,
                                        const float *bias,
                                        float *C, int ldc,
                                        int32_t *c32_tail,
                                        float *max_abs_out);

void qgemm_avxvnni_int32   (int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_avxvnni_int32_k20(int M, int N,
                             const int8_t *A_q, const int8_t *Bp,
                             int32_t *C32, int ldc32);
/* fp16 inout — h_inout_fp16 is the fp16 recurrent
 * storage (read for h_old via VCVTPH2PS, written with new state via
 * VCVTPS2PH); h_out_scratch is fp32 scratch that rnn_fc consumes. */
void qgemm_avxvnni_gru_full_fused_fp16inout(int M, int D,
                                             const int8_t *Xq, int ld_x,
                                             const int8_t *Hq, int ld_h,
                                             const int8_t *Wq_ih,
                                             const int8_t *Wq_hh,
                                             const float *combined_ih,
                                             const float *bias_eff_ih,
                                             const float *combined_hh,
                                             const float *bias_eff_hh,
                                             const float *br_sum,
                                             const float *bz_sum,
                                             const float *bn_i,
                                             const float *bn_h,
                                             uint16_t *h_inout_fp16, int ld_h_inout,
                                             float *h_out_scratch, int ld_h_out);
void qgemm_avx2_gru_full_fused_fp16inout(int M, int D,
                                          const int8_t *Xq, int ld_x,
                                          const int8_t *Hq, int ld_h,
                                          const int8_t *Wq_ih,
                                          const int8_t *Wq_hh,
                                          const float *combined_ih,
                                          const float *bias_eff_ih,
                                          const float *combined_hh,
                                          const float *bias_eff_hh,
                                          const float *br_sum,
                                          const float *bz_sum,
                                          const float *bn_i,
                                          const float *bn_h,
                                          uint16_t *h_inout_fp16, int ld_h_inout,
                                          float *h_out_scratch, int ld_h_out);
void qgemm_avx512vnni_gru_full_fused_fp16inout(int M, int D,
                                                const int8_t *Xq, int ld_x,
                                                const int8_t *Hq, int ld_h,
                                                const int8_t *Wq_ih,
                                                const int8_t *Wq_hh,
                                                const float *combined_ih,
                                                const float *bias_eff_ih,
                                                const float *combined_hh,
                                                const float *bias_eff_hh,
                                                const float *br_sum,
                                                const float *bz_sum,
                                                const float *bn_i,
                                                const float *bn_h,
                                                uint16_t *h_inout_fp16, int ld_h_inout,
                                                float *h_out_scratch, int ld_h_out);
void qgemm_avxvnni_fp32_fused(int M, int N, int K,
                              const int8_t *A_q, const int8_t *Bp,
                              const float *combined_scale, const float *bias,
                              float *C, int ldc, int act_silu,
                              int32_t *c32_tail);
void qgemm_avxvnni_fp32_fused_acc(int M, int N, int K,
                                  const int8_t *A_q, const int8_t *Bp,
                                  const float *combined_scale, const float *bias,
                                  float *C, int ldc, int32_t *c32_tail);
void qgemm_avxvnni_fp32_fused_track_maxabs(int M, int N, int K,
                                           const int8_t *A_q, const int8_t *Bp,
                                           const float *combined_scale,
                                           const float *bias,
                                           float *C, int ldc,
                                           int32_t *c32_tail,
                                           float *max_abs_out);

void qgemm_avx512vnni_int32(int M, int N, int K,
                            const int8_t *A_q, const int8_t *Bp,
                            int32_t *C32, int ldc32);
void qgemm_avx512vnni_int32_k20(int M, int N,
                                const int8_t *A_q, const int8_t *Bp,
                                int32_t *C32, int ldc32);
void qgemm_avx512vnni_fp32_fused(int M, int N, int K,
                                 const int8_t *A_q, const int8_t *Bp,
                                 const float *combined_scale, const float *bias,
                                 float *C, int ldc, int act_silu,
                                 int32_t *c32_tail);
void qgemm_avx512vnni_fp32_fused_acc(int M, int N, int K,
                                     const int8_t *A_q, const int8_t *Bp,
                                     const float *combined_scale, const float *bias,
                                     float *C, int ldc, int32_t *c32_tail);
void qgemm_avx512vnni_fp32_fused_track_maxabs(int M, int N, int K,
                                              const int8_t *A_q, const int8_t *Bp,
                                              const float *combined_scale,
                                              const float *bias,
                                              float *C, int ldc,
                                              int32_t *c32_tail,
                                              float *max_abs_out);

#endif /* x86 */

#ifdef __cplusplus
}
#endif

#endif /* FE_ARCH_KERNELS_H */

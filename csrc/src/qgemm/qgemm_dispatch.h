/*
 * Runtime ISA dispatch for the int8 W8A8 GEMM family. fe_qgemm_init
 * picks the highest-throughput kernel set the host supports.
 *
 * Tier coverage:
 *   ARM: neon (baseline), dotprod (FEAT_DotProd), i8mm (FEAT_I8MM)
 *   x86: avx2 (baseline, FMA3+F16C required), avxvnni, avx512_vnni
 *
 * Pre-AVX2 x86 tiers were previously supported but dropped: FMA3 requires
 * AVX, so a pre-AVX2 dequant epilogue (`bias + scale * c32`) is two
 * roundings (mul, add) vs one (fmadd) on AVX2+. The resulting ~1 ULP/op
 * drift accumulates to ~40 dB SNR across an inference, breaking the
 * cross-tier byte-identity guarantee. F16C is also required for fp16 state
 * conversion. AVX2+FMA3+F16C (Haswell, 2013+) is therefore the x86 minimum.
 */
#ifndef FE_QGEMM_DISPATCH_H
#define FE_QGEMM_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* int32-out GEMM: A[M,K] @ B^T[N,K] -> int32 C32[M,N]. Every tier
 * provides this; dequant/bias/silu runs separately. */
typedef void (*fe_qgemm_int32_fn)(
        int M, int N, int K,
        const int8_t *A_q, const int8_t *Bp,
        int32_t *C32, int ldc32);

/* fp32-out fused: int32 acc -> dequant -> +bias -> optional SiLU ->
 * fp32 store, all in registers. May be NULL on tiers without it. */
typedef void (*fe_qgemm_fp32_fused_fn)(
        int M, int N, int K,
        const int8_t *A_q, const int8_t *Bp,
        const float *combined_scale,        /* scale_in * scales_w[n] */
        const float *bias,                  /* length N, may be NULL */
        float *C, int ldc,
        int act_silu, int32_t *c32_tail);

typedef struct {
    fe_qgemm_int32_fn       gemm_int32;        /* always set */
    fe_qgemm_fp32_fused_fn  gemm_fp32_fused;   /* may be NULL */
    const char             *path_name;
    int                     tier;
} FeQgemmOps;

extern FeQgemmOps fe_qgemm_ops;

/* Idempotent. Returns 0 on success, -1 if no supported SIMD tier
 * exists on the host (no NEON on aarch64, or no AVX2+FMA3+F16C on x86_64). */
int fe_qgemm_init(void);

/* Test hook: force a tier. tier_name is one of: "neon","dotprod","i8mm",
 * "avx2","avxvnni","avx512vnni". -1 if unavailable. */
int  fe_qgemm_force_tier(const char *tier_name);

/* Numeric tier IDs (higher = better). */
enum {
    FE_QGEMM_TIER_NONE            = 0,
    FE_QGEMM_TIER_ARM_NEON        = 10,
    FE_QGEMM_TIER_ARM_DOTPROD        = 20,
    FE_QGEMM_TIER_ARM_I8MM       = 30,
    FE_QGEMM_TIER_X86_AVX2        = 120,
    FE_QGEMM_TIER_X86_AVX_VNNI    = 130,
    FE_QGEMM_TIER_X86_AVX512_VNNI = 140
};

#ifdef __cplusplus
}
#endif

#endif /* FE_QGEMM_DISPATCH_H */

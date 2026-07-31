/*
 * Runtime CPU detection + FeQgemmOps population. Called once from
 * fe_init; thereafter the inference path indirect-calls through the
 * table at M-block granularity.
 *
 * Compiled without -march so the detector cannot SIGILL on older
 * hardware; per-tier kernel TUs get their own -march via CMake.
 *
 * No scalar runtime tier: if no SIMD tier matches, fe_qgemm_init
 * returns -1 and fe_init propagates failure to the caller.
 */
#include "qgemm_dispatch.h"
#include "arch_kernels.h"
#include "cpu_detect.h"
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

FeQgemmOps fe_qgemm_ops;

/* Unreachable in production: every GEMM is tile-aligned (M%MR==0, N%NR==0),
 * enforced by the _Static_assert block in fe_config_medium.h. A non-tile GEMM
 * would otherwise need the removed scalar remainder kernel — abort loudly rather
 * than silently drop the M%MR / N%NR strip. Unsupported ISAs, by contrast,
 * fail at fe_init via a non-zero return (qgemm_dispatch has no scalar
 * runtime tier). */
void fe_qgemm_tail_unsupported(void) {
    fprintf(stderr,
            "fe_qgemm: GEMM dimension is not a multiple of the %dx%d tile; "
            "the scalar remainder kernel was removed. Aborting.\n",
            FE_QGEMM_MR, FE_QGEMM_NR);
    abort();
}

static int g_qgemm_initialized = 0;
static int g_qgemm_force_tier  = -1;   /* -1 = auto-pick */

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

static void fill_ops_neon(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_neon_int32;
    ops->gemm_fp32_fused  = qgemm_neon_fp32_fused;
    ops->path_name        = "neon";
    ops->tier             = FE_QGEMM_TIER_ARM_NEON;
}

static void fill_ops_dotprod(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_dotprod_int32;
    ops->gemm_fp32_fused  = qgemm_dotprod_fp32_fused;
    ops->path_name        = "dotprod";
    ops->tier             = FE_QGEMM_TIER_ARM_DOTPROD;
}

static void fill_ops_i8mm(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_i8mm_int32;
    ops->gemm_fp32_fused  = qgemm_i8mm_fp32_fused;
    ops->path_name        = "i8mm";
    ops->tier             = FE_QGEMM_TIER_ARM_I8MM;
}

#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

static void fill_ops_avx2(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_avx2_int32;
    ops->gemm_fp32_fused  = qgemm_avx2_fp32_fused;
    ops->path_name        = "avx2";
    ops->tier             = FE_QGEMM_TIER_X86_AVX2;
}

static void fill_ops_avxvnni(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_avxvnni_int32;
    ops->gemm_fp32_fused  = qgemm_avxvnni_fp32_fused;
    ops->path_name        = "avxvnni";
    ops->tier             = FE_QGEMM_TIER_X86_AVX_VNNI;
}

static void fill_ops_avx512vnni(FeQgemmOps *ops) {
    memset(ops, 0, sizeof(*ops));
    ops->gemm_int32       = qgemm_avx512vnni_int32;
    ops->gemm_fp32_fused  = qgemm_avx512vnni_fp32_fused;
    ops->path_name        = "avx512vnni";
    ops->tier             = FE_QGEMM_TIER_X86_AVX512_VNNI;
}

#endif

/* Pick the highest-tier ops table the host supports, optionally
 * clamped to a forced tier. Leaves ops->tier == NONE if nothing matched. */
static void select_ops(FeQgemmOps *ops, int forced_tier) {
    memset(ops, 0, sizeof(*ops));

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    uint32_t caps = fe_cpu_arm_caps();
    if (caps & FE_ARM_HAS_NEON) fill_ops_neon(ops);
    if (forced_tier == FE_QGEMM_TIER_ARM_NEON) return;
    if (caps & FE_ARM_HAS_DOTPROD) fill_ops_dotprod(ops);
    if (forced_tier == FE_QGEMM_TIER_ARM_DOTPROD) return;
    if (caps & FE_ARM_HAS_I8MM) fill_ops_i8mm(ops);
    if (forced_tier == FE_QGEMM_TIER_ARM_I8MM) return;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    uint32_t caps = fe_cpu_x86_caps();
    /* AVX2+FMA3+F16C is the minimum supported tier on x86. Pre-AVX2 tiers
     * were dropped because their lack of FMA3 breaks cross-tier
     * byte-identity with AVX2+, and F16C is required for fp16 state. */
    if ((caps & FE_X86_HAS_AVX2) && (caps & FE_X86_HAS_OS_AVX))
        fill_ops_avx2(ops);
    if (forced_tier == FE_QGEMM_TIER_X86_AVX2) return;
    if ((caps & FE_X86_HAS_AVXVNNI) && (caps & FE_X86_HAS_OS_AVX))
        fill_ops_avxvnni(ops);
    if (forced_tier == FE_QGEMM_TIER_X86_AVX_VNNI) return;
    if ((caps & FE_X86_HAS_AVX512VNNI) && (caps & FE_X86_HAS_OS_AVX512))
        fill_ops_avx512vnni(ops);
    if (forced_tier == FE_QGEMM_TIER_X86_AVX512_VNNI) return;
#endif
}

int fe_qgemm_init(void) {
    if (g_qgemm_initialized && g_qgemm_force_tier < 0) return 0;
    select_ops(&fe_qgemm_ops, g_qgemm_force_tier);
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_NONE) {
        fprintf(stderr,
                "[fe_qgemm] FATAL: host CPU has no supported SIMD tier.\n"
                "  required: ARM NEON (arm64) or x86 AVX2+FMA3+F16C (x86_64) at minimum.\n"
                "  cpu=%s\n", fe_cpu_brand());
        return -1;
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* route the activation quantize/min-max to the zmm-native path
     * only when the AVX-512 tier is selected (CPU genuinely has AVX-512 →
     * no SIGILL elsewhere). Keeps the 16-wide GEMM from being fed by an
     * 8-wide quantize. */
    {
        extern int fe_qg_x86_avx512;
        fe_qg_x86_avx512 = (fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI);
    }
    /* pre-fault the AVX2 vpmovsxbw+vpmaddwd path's BSS scratch pages so
     * first-touch faults don't leak into steady-state percentiles. Always
     * called regardless of selected tier — the AVX2 buffers are tier-static
     * and the one-time init cost is outside steady-state timing. */
    qgemm_avx2_prefault_buffers();
#endif
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    /* pre-fault the I8MM row-pair pre-pack BSS scratch pages. Same
     * move-first-touch-faults-out-of-steady-state rationale as the AVX2
     * prefault above. */
    qgemm_arm_i8mm_prefault_buffers();
    /* pre-fault DOTPROD row-quad pre-pack BSS scratch pages (same rationale). */
    qgemm_arm_dotprod_prefault_buffers();
#endif
    g_qgemm_initialized = 1;
#ifndef FE_QGEMM_QUIET
    static int announced_for_path = -1;
    if (announced_for_path != fe_qgemm_ops.tier) {
        fprintf(stderr, "[fe_qgemm] cpu=%s tier=%s\n",
                fe_cpu_brand(), fe_qgemm_ops.path_name);
        announced_for_path = fe_qgemm_ops.tier;
    }
#endif
    return 0;
}

/* Force a tier on the next fe_qgemm_init (test harness only). NULL
 * clears the override. Returns -1 if the tier is not available. */
int fe_qgemm_force_tier(const char *tier_name) {
    if (!tier_name) {
        g_qgemm_force_tier  = -1;
        g_qgemm_initialized = 0;
        return 0;
    }
    int tier = -1;
    if      (!strcmp(tier_name, "neon"))       tier = FE_QGEMM_TIER_ARM_NEON;
    else if (!strcmp(tier_name, "dotprod"))       tier = FE_QGEMM_TIER_ARM_DOTPROD;
    else if (!strcmp(tier_name, "i8mm"))      tier = FE_QGEMM_TIER_ARM_I8MM;
    else if (!strcmp(tier_name, "avx2"))       tier = FE_QGEMM_TIER_X86_AVX2;
    else if (!strcmp(tier_name, "avxvnni"))    tier = FE_QGEMM_TIER_X86_AVX_VNNI;
    else if (!strcmp(tier_name, "avx512vnni")) tier = FE_QGEMM_TIER_X86_AVX512_VNNI;
    else return -1;

    FeQgemmOps probe;
    select_ops(&probe, tier);
    if (probe.tier != tier) return -1;

    g_qgemm_force_tier  = tier;
    g_qgemm_initialized = 0;
    return 0;
}

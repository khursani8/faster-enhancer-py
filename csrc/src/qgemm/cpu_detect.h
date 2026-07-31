/*
 * cpu_detect.h -- Internal: runtime CPU capability detection.
 *
 * Single TU per arch is responsible for populating these bits. The TU MUST
 * be compiled with conservative -march flags (no DOTPROD/I8MM/AVX-512), and
 * never inlined into anything that runs before fe_qgemm_init() -- otherwise
 * the detector itself can SIGILL on the very CPUs it's trying to detect.
 *
 * Implementations:
 *   src/qgemm/arm/cpu_arm.c   (HWCAP via getauxval on Linux/Android, sysctl
 *                              on macOS/iOS, IsProcessorFeaturePresent on
 *                              Windows ARM64)
 *   src/qgemm/x86/cpu_x86.c   (__cpuid_count + xgetbv to confirm OS XSAVE)
 */
#ifndef FE_CPU_DETECT_H
#define FE_CPU_DETECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ARM runtime tiers. */
enum {
    FE_ARM_HAS_NEON      = 1u << 0,   /* always set on arm64 */
    FE_ARM_HAS_DOTPROD   = 1u << 1,   /* FEAT_DotProd (vdotq_s32)         */
    FE_ARM_HAS_I8MM      = 1u << 2    /* FEAT_I8MM   (vmmlaq_s32 / I8MM) */
};
uint32_t fe_cpu_arm_caps(void);

/* x86 runtime tiers plus OS vector-state support. FE_X86_HAS_AVX2 implies
 * the AVX2+FMA3+F16C floor required by the kernels and fp16 state storage;
 * FE_X86_HAS_AVX512VNNI implies that floor plus AVX-512F/BW/VL/VNNI. */
enum {
    FE_X86_HAS_AVX2       = 1u << 0,
    FE_X86_HAS_AVXVNNI    = 1u << 1,  /* Alder Lake+, Zen 4+ (256-bit VNNI) */
    FE_X86_HAS_AVX512VNNI = 1u << 2,
    FE_X86_HAS_OS_AVX     = 1u << 3,  /* xgetbv confirmed OS preserves YMM */
    FE_X86_HAS_OS_AVX512  = 1u << 4   /* xgetbv confirmed OS preserves ZMM */
};
uint32_t fe_cpu_x86_caps(void);

/* Human-readable name of the host (printed at init for diagnostics). */
const char *fe_cpu_brand(void);

#ifdef __cplusplus
}
#endif

#endif /* FE_CPU_DETECT_H */

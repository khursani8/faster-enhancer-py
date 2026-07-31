/*
 * cpu_arm.c -- Runtime ARM capability detection.
 *
 * MUST be compiled with conservative flags only (no -march=armv8.6-a+i8mm
 * at file scope) to avoid the compiler emitting tier-specific instructions
 * in this detector. CMakeLists keeps this TU at baseline flags.
 *
 * Sources:
 *   Linux / Android : getauxval(AT_HWCAP / AT_HWCAP2) bitmasks
 *   macOS / iOS     : sysctlbyname("hw.optional.arm.FEAT_*")
 *   Windows ARM64   : IsProcessorFeaturePresent
 *   else            : compile-time __ARM_FEATURE_* fallback (assumes
 *                     the build target's features are present at runtime)
 */
#include "../cpu_detect.h"

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

#if defined(__linux__) || defined(__ANDROID__)
#  include <sys/auxv.h>
#  ifndef AT_HWCAP
#    define AT_HWCAP  16
#  endif
#  ifndef AT_HWCAP2
#    define AT_HWCAP2 26
#  endif
#  ifndef HWCAP_ASIMDDP
#    define HWCAP_ASIMDDP   (1UL << 20)
#  endif
#  ifndef HWCAP2_I8MM
#    define HWCAP2_I8MM     (1UL << 13)
#  endif
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <stdint.h>
static int fe_sysctl_bool(const char *name) {
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) return 0;
    return v ? 1 : 0;
}
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#    define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#  endif
#endif

uint32_t fe_cpu_arm_caps(void) {
    uint32_t caps = FE_ARM_HAS_NEON;   /* arm64 always has NEON */

#if defined(__linux__) || defined(__ANDROID__)
    unsigned long h1 = getauxval(AT_HWCAP);
    unsigned long h2 = getauxval(AT_HWCAP2);
    if (h1 & HWCAP_ASIMDDP) caps |= FE_ARM_HAS_DOTPROD;
    if (h2 & HWCAP2_I8MM)   caps |= FE_ARM_HAS_I8MM;

#elif defined(__APPLE__)
    if (fe_sysctl_bool("hw.optional.arm.FEAT_DotProd"))     caps |= FE_ARM_HAS_DOTPROD;
    if (fe_sysctl_bool("hw.optional.arm.FEAT_I8MM"))        caps |= FE_ARM_HAS_I8MM;

#elif defined(_WIN32)
    if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE))
        caps |= FE_ARM_HAS_DOTPROD;
    /* Windows on ARM does not expose a portable FEAT_I8MM PF_ constant here;
     * fall back to compile-time macro if our build target requires it. */
#  if defined(__ARM_FEATURE_MATMUL_INT8)
    caps |= FE_ARM_HAS_I8MM;
#  endif

#else
    /* Unknown OS -- trust compile-time macros. Safe because the binary was
     * built with these features assumed. */
#  if defined(__ARM_FEATURE_DOTPROD)
    caps |= FE_ARM_HAS_DOTPROD;
#  endif
#  if defined(__ARM_FEATURE_MATMUL_INT8)
    caps |= FE_ARM_HAS_I8MM;
#  endif
#endif
    return caps;
}

const char *fe_cpu_brand(void) {
#if defined(__APPLE__)
    static char buf[128];
    size_t sz = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, NULL, 0) == 0)
        return buf;
    return "Apple ARM64";
#else
    return "ARM64";
#endif
}

#else /* not aarch64 -- provide stubs so dispatch.c links cleanly. */
uint32_t fe_cpu_arm_caps(void) { return 0; }
const char *fe_cpu_brand(void) { return "non-ARM"; }
#endif

/*
 * cpu_x86.c -- Runtime x86 capability detection via CPUID + XGETBV.
 *
 * Compiled WITHOUT -mavx2 / -mavxvnni / -mavx512* so the detector itself
 * cannot SIGILL on older CPUs.
 */
#include "../cpu_detect.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#if defined(_MSC_VER)
#  include <intrin.h>
static inline void fe_cpuid_count(int leaf, int sub,
                                  uint32_t *a, uint32_t *b,
                                  uint32_t *c, uint32_t *d) {
    int regs[4];
    __cpuidex(regs, leaf, sub);
    *a = regs[0]; *b = regs[1]; *c = regs[2]; *d = regs[3];
}
static inline uint64_t fe_xgetbv0(void) { return _xgetbv(0); }
#else
#  include <cpuid.h>
static inline void fe_cpuid_count(int leaf, int sub,
                                  uint32_t *a, uint32_t *b,
                                  uint32_t *c, uint32_t *d) {
    __cpuid_count(leaf, sub, *a, *b, *c, *d);
}
static inline uint64_t fe_xgetbv0(void) {
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}
#endif

uint32_t fe_cpu_x86_caps(void) {
    uint32_t caps = 0;
    uint32_t a, b, c, d;

    /* Leaf 1 -- AVX, FMA, F16C, OSXSAVE. */
    fe_cpuid_count(1, 0, &a, &b, &c, &d);
    int has_fma     = (c >> 12) & 1;
    int has_f16c    = (c >> 29) & 1;
    int has_avx     = (c >> 28) & 1;
    int has_osxsave = (c >> 27) & 1;

    /* OS XSAVE check: confirms the OS preserves YMM (bit 2) / ZMM (bits 5..7)
     * across context switches. Without this, AVX instructions raise #UD even
     * on a capable CPU. */
    int os_ymm = 0, os_zmm = 0;
    if (has_osxsave) {
        uint64_t xcr0 = fe_xgetbv0();
        os_ymm = ((xcr0 & 0x6) == 0x6);                  /* XMM + YMM state */
        os_zmm = ((xcr0 & 0xE6) == 0xE6);                 /* + ZMM_lo, ZMM_hi, opmask */
    }
    if (has_avx && os_ymm) caps |= FE_X86_HAS_OS_AVX;
    if (os_zmm) caps |= FE_X86_HAS_OS_AVX512;

    /* Leaf 7 sub 0 -- AVX2, AVX-512 family. */
    fe_cpuid_count(7, 0, &a, &b, &c, &d);
    int has_avx2        = (b >> 5)  & 1;
    int has_avx512f     = (b >> 16) & 1;
    int has_avx512bw    = (b >> 30) & 1;
    int has_avx512vl    = (b >> 31) & 1;
    int has_avx512vnni  = (c >> 11) & 1;
    if (has_avx2 && has_fma && has_f16c) caps |= FE_X86_HAS_AVX2;
    if (has_avx2 && has_fma && has_f16c && has_avx512f && has_avx512bw
        && has_avx512vl && has_avx512vnni) {
        caps |= FE_X86_HAS_AVX512VNNI;
    }

    /* Leaf 7 sub 1 -- AVX-VNNI (Alder Lake+, Zen4+). */
    fe_cpuid_count(7, 1, &a, &b, &c, &d);
    if (has_avx2 && has_fma && has_f16c && ((a >> 4) & 1))
        caps |= FE_X86_HAS_AVXVNNI;

    /* Tiers also require the OS to preserve their vector state. */
    if (!(caps & FE_X86_HAS_OS_AVX512)) {
        caps &= ~FE_X86_HAS_AVX512VNNI;
    }
    if (!(caps & FE_X86_HAS_OS_AVX)) {
        caps &= ~(FE_X86_HAS_AVX2 | FE_X86_HAS_AVXVNNI);
    }
    return caps;
}

const char *fe_cpu_brand(void) {
    static char buf[64];
    uint32_t a, b, c, d;
    /* CPUID leaf 0x80000002..0x80000004 -> 48-char brand string. */
    fe_cpuid_count(0x80000000, 0, &a, &b, &c, &d);
    if (a < 0x80000004) return "x86_64";
    uint32_t *p = (uint32_t *)buf;
    fe_cpuid_count(0x80000002, 0, &p[0],  &p[1],  &p[2],  &p[3]);
    fe_cpuid_count(0x80000003, 0, &p[4],  &p[5],  &p[6],  &p[7]);
    fe_cpuid_count(0x80000004, 0, &p[8],  &p[9],  &p[10], &p[11]);
    buf[48] = 0;
    /* Strip leading whitespace. */
    char *s = buf;
    while (*s == ' ') ++s;
    return s;
}

#else /* not x86 -- stub */
uint32_t fe_cpu_x86_caps(void) { return 0; }
const char *fe_cpu_brand(void) { return "non-x86"; }
#endif

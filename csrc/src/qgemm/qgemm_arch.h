// Compile-time architecture detection for the int8 GEMM family.
// Each FE_QGEMM_HAVE_* flag mirrors a __ARM_FEATURE_* / __AVX*__ predicate
// and is set per-TU based on that TU's -march flag. The dispatcher uses
// runtime detection (qgemm_dispatch.c); these flags are for the kernel
// TUs themselves to gate intrinsic usage.
#ifndef FE_QGEMM_ARCH_H
#define FE_QGEMM_ARCH_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
  #include <arm_neon.h>
  #define FE_QGEMM_NEON 1
  #if defined(__ARM_FEATURE_DOTPROD)
    #define FE_QGEMM_HAVE_DOTPROD 1
  #endif
  #if defined(__ARM_FEATURE_MATMUL_INT8)
    #define FE_QGEMM_HAVE_I8MM 1
  #endif
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define FE_QGEMM_X86 1
  /* AVX2+FMA3+F16C is the minimum supported tier on x86. FMA3 is required
   * for cross-tier bit-identity and F16C for fp16 state conversion. */
  #if defined(__AVX512VNNI__)
    #include <immintrin.h>
    #define FE_QGEMM_HAVE_AVX512_VNNI 1
  #elif defined(__AVXVNNI__)
    #include <immintrin.h>
    #define FE_QGEMM_HAVE_AVXVNNI 1
  #elif defined(__AVX2__)
    #include <immintrin.h>
    #define FE_QGEMM_HAVE_AVX2 1
  #endif
#endif

// Static scratch K bound. Current Medium runtime max K is 288 (C1=96, k=3);
// 384 leaves deliberate headroom without changing hot-path control flow.
#ifndef FE_QGEMM_MAX_K
#define FE_QGEMM_MAX_K 384
#endif

// Static GRU hidden-size bound. Current Medium D is 72; 128 leaves deliberate
// headroom and keeps the fused kernels free of VLAs.
#ifndef FE_QGEMM_MAX_GRU_D
#define FE_QGEMM_MAX_GRU_D 128
#endif

// 3 * MAX_GRU_D -- bounds the per-call ih_tile scratch in gru_full_fused
// kernels (x86 fall-back tiers). Single-threaded so static-local is safe.
#ifndef FE_QGEMM_MAX_GRU_D3
#define FE_QGEMM_MAX_GRU_D3 (3 * FE_QGEMM_MAX_GRU_D)
#endif

// Cache-line alignment for hot static scratch buffers. The kernels use
// unaligned loads (loadu), but aligning the backing store to 64 B removes
// cache-line-split loads/stores at block boundaries. Pure layout — bit-id
// unaffected (alignment changes no value).
#ifndef FE_ALIGN64
#  if defined(_MSC_VER)
#    define FE_ALIGN64 __declspec(align(64))
#  else
#    define FE_ALIGN64 __attribute__((aligned(64)))
#  endif
#endif

// Max GRU microkernel row-tile across tiers. ARM uses MR=8; x86 uses
// MR=12 (12 vpdpbusd accumulator chains saturate the lat-5/tput-2 unit).
// Bounds the shared r/z band scratch so MR=12 row indices never overflow.
#ifndef FE_QGEMM_GRU_MR
#define FE_QGEMM_GRU_MR 12
#endif

#endif // FE_QGEMM_ARCH_H

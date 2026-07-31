/* ARM AdvSIMD (NEON) FFT TU. NEON is baseline on arm64, no per-file flag needed. */
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include "fft_arch.h"
#include "fft_neon.inl"
#endif

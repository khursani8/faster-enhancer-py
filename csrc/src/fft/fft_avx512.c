/* x86 AVX-512 FFT TU. Compiled with AVX-512 family flags per-file via CMake. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "fft_arch.h"
#include "fft_avx512.inl"
#endif

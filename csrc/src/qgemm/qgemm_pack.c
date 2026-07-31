// Quantize a row-major fp32 weight matrix [N, K] into per-output-channel
// symmetric int8 + per-row fp32 scale, packed into the DOTPROD-friendly layout
// consumed by every SIMD tier (block_n x k4_group x lane=NR x k_lane=4).
//
// The I8MM tier needs a different layout (k8_group x col_pair x col x k8);
// fe_qgemm_repack_i8mm() converts a block in place once the dispatcher has
// confirmed I8MM is the chosen path at runtime.
#include "fe_qgemm.h"
#include "qgemm_arch.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

// Convert one packed N-block from DOTPROD layout to I8MM layout. Both
// layouts occupy K*NR bytes; only the within-block ordering changes
// (I8MM wants [k8_group][col_pair=4][col=2][k=8]). Uses baseline NEON,
// no I8MM needed. Non-aarch64 builds expose a stub so callers in shared
// code (e.g. fe_attention.c) link cleanly -- the runtime tier check
// guarantees the function is never actually called on non-aarch64.
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#include <arm_neon.h>
void fe_qgemm_i8mm_repack_block(const int8_t *src, int K, int8_t *dst) {
    int k8_groups = K / 8;
    for (int g8 = 0; g8 < k8_groups; ++g8) {
        const int8_t *src_lo = src + (size_t)(2 * g8)     * 32;
        const int8_t *src_hi = src + (size_t)(2 * g8 + 1) * 32;
        int8_t *out = dst + (size_t)g8 * 64;
        int32x4_t g0_lo = vld1q_s32((const int32_t *)(src_lo +  0));
        int32x4_t g0_hi = vld1q_s32((const int32_t *)(src_lo + 16));
        int32x4_t g1_lo = vld1q_s32((const int32_t *)(src_hi +  0));
        int32x4_t g1_hi = vld1q_s32((const int32_t *)(src_hi + 16));
        vst1q_s32((int32_t *)(out +  0), vzip1q_s32(g0_lo, g1_lo));
        vst1q_s32((int32_t *)(out + 16), vzip2q_s32(g0_lo, g1_lo));
        vst1q_s32((int32_t *)(out + 32), vzip1q_s32(g0_hi, g1_hi));
        vst1q_s32((int32_t *)(out + 48), vzip2q_s32(g0_hi, g1_hi));
    }
}
#else
void fe_qgemm_i8mm_repack_block(const int8_t *src, int K, int8_t *dst) {
    (void)src; (void)K; (void)dst;
}
#endif

/* Walk the DOTPROD-packed weight buffer and accumulate the int8 row sums
 * Σ_k W_i8[n, k] (n in original output-channel order). The packed layout
 * is [block_n][k4_group][lane=NR][k_lane=4]; lanes past N within a partial
 * block were zeroed by the packer, so row_sums for those don't matter
 * (they're never indexed by a valid output channel). Padding K (k_lane >
 * k_max) was also zeroed. Cheap O(N*K) integer accumulation. */
void fe_qgemm_compute_row_sums(const int8_t *packed, int N, int K,
                               int32_t *row_sums_out) {
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    int n_blocks  = (N + NR - 1) / NR;
    for (int n = 0; n < N; ++n) row_sums_out[n] = 0;
    for (int block_n = 0; block_n < n_blocks; ++block_n) {
        const int8_t *blk = packed
                          + (size_t)block_n * (size_t)k4_groups * NR * 4;
        for (int lane = 0; lane < NR; ++lane) {
            int n_global = block_n * NR + lane;
            if (n_global >= N) break;
            int32_t s = 0;
            for (int g = 0; g < k4_groups; ++g) {
                const int8_t *q = blk + (size_t)g * NR * 4 + (size_t)lane * 4;
                s += (int32_t)q[0] + (int32_t)q[1]
                   + (int32_t)q[2] + (int32_t)q[3];
            }
            row_sums_out[n_global] = s;
        }
    }
}

void fe_qgemm_pack_W(const float *W_fp32, int N, int K,
                     int8_t *packed_out, float *scales_out) {
    const int NR = FE_QGEMM_NR;

    // Per-channel scale = max|W[n,:]| / 127. Zero rows get scale=1 to avoid div0.
    for (int n = 0; n < N; ++n) {
        float m = 0.0f;
        const float *row = W_fp32 + (size_t)n * K;
        for (int k = 0; k < K; ++k) {
            float a = fabsf(row[k]);
            if (a > m) m = a;
        }
        scales_out[n] = (m == 0.0f) ? 1.0f : (m / 127.0f);
    }

    // DOTPROD layout: [block_n][k4_group][lane][k_lane]. AVX-VNNI / AVX-512 VNNI
    // also consume this directly; AVX2 / NEON-baseline widen on the fly.
    int k4_groups = (K + 3) / 4;
    int n_blocks  = (N + NR - 1) / NR;
    for (int block_n = 0; block_n < n_blocks; ++block_n) {
        int8_t *out = packed_out + (size_t)block_n * (size_t)k4_groups * NR * 4;
        for (int g = 0; g < k4_groups; ++g) {
            int k_base = g * 4;
            for (int lane = 0; lane < NR; ++lane) {
                int n_global = block_n * NR + lane;
                int8_t *dst = out + (size_t)g * NR * 4 + (size_t)lane * 4;
                for (int kk = 0; kk < 4; ++kk) {
                    int k = k_base + kk;
                    int8_t q = 0;
                    if (n_global < N && k < K) {
                        float val = W_fp32[(size_t)n_global * K + k];
                        int qi = (int)lroundf(val / scales_out[n_global]);
                        /* Lower clamp -127 (not -128) is a cross-tier byte-id /
                         * saturation-margin invariant: the symmetric range keeps
                         * the i16-accumulation headroom and avoids the -128
                         * asymmetry, so packed weights match across all kernel
                         * tiers. It is also load-bearing for the AVX2 vpsignb
                         * path (cannot negate -128: i8 wraps to -128, wrong
                         * sign). That path is NOT dead -- it is the K-odd /
                         * large-MK fallback in qgemm_avx2.c; even-K shapes take
                         * the i16 (vpmovsxbw+vpmaddwd) primary path, so it is
                         * rarely exercised but stays correct only via this clamp. */
                        if (qi < -127) qi = -127;
                        if (qi >  127) qi =  127;
                        q = (int8_t)qi;
                    }
                    dst[kk] = q;
                }
            }
        }
    }
}

// In-place DOTPROD -> I8MM layout conversion for one full N-block. Both layouts
// occupy K*NR bytes so the transform is shape-preserving (uses a small
// stack scratch). Only valid when K is a multiple of 8. Current production
// shapes keep N a multiple of NR, so every dispatched N-block is full.
//
// Called by the dispatcher when I8MM is the chosen tier at runtime -- NOT
// from fe_qgemm_pack_W, which has no knowledge of the runtime tier choice.
void fe_qgemm_repack_i8mm(int8_t *packed, int N, int K) {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    if ((K & 7) != 0) return;
    const int NR = FE_QGEMM_NR;
    int k4_groups = (K + 3) / 4;
    int full_blocks = N / NR;
    int8_t scratch[FE_QGEMM_MAX_K * 8];
    for (int bn = 0; bn < full_blocks; ++bn) {
        int8_t *blk = packed + (size_t)bn * (size_t)k4_groups * NR * 4;
        fe_qgemm_i8mm_repack_block(blk, K, scratch);
        memcpy(blk, scratch, (size_t)(K / 8) * 64);
    }
#else
    (void)packed; (void)N; (void)K;
#endif
}

/*
 * qgemm_quant.c -- Per-tensor ASYMMETRIC uint8 activation quantisation.
 *
 *   fe_quantize_activation             : dynamic, scans min+max first
 *   fe_quantize_activation_with_scale  : caller supplies (scale, zp)
 *
 * Asymmetric semantics:
 *   x_f ~ scale * (q_u8 - zp_u8),   q_u8 ∈ [0, 255],  zp_u8 ∈ [0, 255]
 *
 * Storage: we write int8 = q_u8 - 128 so the existing signedxsigned GEMM
 * kernels (DOTPROD/I8MM/AVX2/AVX-VNNI/AVX-512-VNNI) consume the buffer
 * unchanged. The wrapper layer accounts for the 128-bias and the zero-
 * point in the dequant epilogue via:
 *   y[m,n] = scale*scale_w[n] * (C32[m,n] + (128-zp_u8)*row_sums[n]) + bias[n]
 * (cf. fe_qgemm.c). Because the AVX-VNNI kernel ALREADY does its own
 * internal XOR-128 + wsum subtraction to satisfy VPDPBUSD's u8xi8 spec,
 * passing it our shifted int8 buffer reproduces the correct C32 (signed
 * matmul) and the wrapper-side correction lands the asymmetric semantics.
 */
#include "fe_qgemm.h"
#include "fe_simd.h"
#include "fe_fp16.h"
#include "qgemm_arch.h"
#include "qgemm_simd_post.inl"
#include <math.h>

/* SIMD asymmetric u8 quantize: q_i8 = clamp(round(x*inv) + zp_off, -127, 127),
 * where zp_off = zp_u8 - 128. Lower bound is -127 (not -128) as a CROSS-TIER
 * byte-id / saturation-margin invariant: keeping the symmetric [-127,127] range
 * preserves the i16-accumulation headroom and avoids the -128 asymmetry, so the
 * quantized bytes are identical across every kernel tier (DOTPROD / I8MM / AVX2
 * / VNNI). It is also load-bearing for the AVX2 vpsignb path: vpsignb cannot
 * negate -128 (it wraps to -128 with the wrong sign and corrupts a*b when a=-128
 * AND b=-128 share a pair lane). That path is NOT dead -- it is the K-odd /
 * large-MK fallback (the `else` branch of qgemm_avx2_int32 in x86/qgemm_avx2.c).
 * This model's even-K shapes take the i16 (vpmovsxbw+vpmaddwd) primary path, so
 * the fallback is rarely if ever exercised, but it stays correct only via this
 * clamp. Cost is ~1 LSB of precision on the most-negative bin. */
static inline void fe_qg_quantize_asym(const float *p, int n,
                                       int8_t *out, float inv,
                                       int32_t zp_off) {
    int i = 0;
    const float zof = (float)zp_off;

/* NB: no inline FE_QPOST_AVX512 branch — this TU is compiled -mavx2 so it
 * would never compile. The AVX-512 tier reaches a zmm quantize via the
 * target-clone fe_qg_quantize_asym_z (runtime-gated on fe_qg_x86_avx512). */
#if defined(FE_QPOST_AVX2)
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 vzof = _mm256_set1_ps(zof);
    const __m128i vmin8 = _mm_set1_epi8(-127);
    for (; i + 7 < n; i += 8) {
        __m256i q = _mm256_cvtps_epi32(
            _mm256_fmadd_ps(_mm256_loadu_ps(p + i), vinv, vzof));
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i s  = _mm_packs_epi32(lo, hi);
        __m128i b  = _mm_packs_epi16(s, s);
        b = _mm_max_epi8(b, vmin8);
        _mm_storel_epi64((__m128i *)(out + i), b);
    }
#elif defined(FE_QGEMM_NEON)
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vzof = vdupq_n_f32(zof);
    const int8x16_t vmin8q = vdupq_n_s8(-127);
    const int8x8_t  vmin8h = vdup_n_s8(-127);
    for (; i + 15 < n; i += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vfmaq_f32(vzof, vld1q_f32(p + i +  0), vinv));
        int32x4_t q1 = vcvtnq_s32_f32(vfmaq_f32(vzof, vld1q_f32(p + i +  4), vinv));
        int32x4_t q2 = vcvtnq_s32_f32(vfmaq_f32(vzof, vld1q_f32(p + i +  8), vinv));
        int32x4_t q3 = vcvtnq_s32_f32(vfmaq_f32(vzof, vld1q_f32(p + i + 12), vinv));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        int8x16_t b = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        b = vmaxq_s8(b, vmin8q);
        vst1q_s8(out + i, b);
    }
    for (; i + 3 < n; i += 4) {
        int32x4_t q = vcvtnq_s32_f32(vfmaq_f32(vzof, vld1q_f32(p + i), vinv));
        int16x4_t s = vqmovn_s32(q);
        int8x8_t  b = vqmovn_s16(vcombine_s16(s, vdup_n_s16(0)));
        b = vmax_s8(b, vmin8h);
        uint32_t  w = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b), 0);
        for (int k = 0; k < 4; ++k) out[i + k] = (int8_t)((w >> (k * 8)) & 0xff);
    }
#endif

    for (; i < n; ++i) {
        int q = (int)lroundf(p[i] * inv) + zp_off;
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
}

/* Pick (scale, zp_u8) from a [min, max] interval, honouring the symmetric
 * pre-condition that zero stays representable (so bias / pad / masked
 * activations all map to a single integer code). Standard QNNPACK math:
 *   scale = max((max - min) / 255, eps)
 *   zp_u8 = round(-min / scale), clamped to [0, 255]
 * The (max - min) span is rescued from zero by eps so a constant input
 * still produces a usable codebook. */
static inline FeActQuant pick_qparams(float vmin, float vmax) {
    /* Make zero representable: extend the range to include 0. */
    if (vmin > 0.0f) vmin = 0.0f;
    if (vmax < 0.0f) vmax = 0.0f;
    float span = vmax - vmin;
    if (span < 1e-12f) span = 1e-12f;
    float scale = span / 255.0f;
    float zp_f  = -vmin / scale;
    int   zp    = (int)lroundf(zp_f);
    if (zp < 0)   zp = 0;
    if (zp > 255) zp = 255;
    FeActQuant q;
    q.scale = scale;
    q.zp    = zp;
#ifdef FE_DEBUG_FORCE_SYMMETRIC
    /* Debug: collapse to symmetric (max-abs/127, zp=128) to bisect. */
    {
        float m = vmax > -vmin ? vmax : -vmin;
        if (m == 0.0f) m = 1.0f;
        q.scale = m / 127.0f;
        q.zp    = 128;
    }
#endif
    return q;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
/* zmm-native activation quantize for the AVX-512 tier. This TU is
 * compiled -mavx2 (it must run on the lowest x86 tier), so a plain
 * FE_QPOST_AVX512 branch could never compile here — without these clones
 * the AVX-512 tier quantized at 256-bit while the GEMM ran 512-bit. These
 * target-attribute clones get
 * AVX-512 codegen inside the -mavx2 TU and run ONLY when fe_qg_x86_avx512
 * is set (tier==AVX-512, i.e. the CPU genuinely has AVX-512 → no SIGILL on
 * AVX2 hosts where the flag stays 0). Bit-id: min/max is exact +
 * order-invariant; quantize is per-element fmadd→cvtps_epi32(round-nearest)
 * →cvtsepi32_epi8(saturate)→max(-127), byte-identical to the AVX2 path. */
int fe_qg_x86_avx512 = 0;

__attribute__((target("avx512f,avx512bw,avx512vl")))
static void fe_qg_min_max_z(const float *p, int n, float *min_out, float *max_out) {
    float vmin = 0.0f, vmax = 0.0f;
    if (n > 0) { vmin = p[0]; vmax = p[0]; }
    int i = 0;
    if (n >= 16) {
        __m512 vmn = _mm512_set1_ps(p[0]), vmx = vmn;
        for (; i + 15 < n; i += 16) {
            __m512 v = _mm512_loadu_ps(p + i);
            vmn = _mm512_min_ps(vmn, v);
            vmx = _mm512_max_ps(vmx, v);
        }
        vmin = _mm512_reduce_min_ps(vmn);
        vmax = _mm512_reduce_max_ps(vmx);
    }
    for (; i < n; ++i) {
        if (p[i] < vmin) vmin = p[i];
        if (p[i] > vmax) vmax = p[i];
    }
    *min_out = vmin; *max_out = vmax;
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
static void fe_qg_quantize_asym_z(const float *p, int n, int8_t *out,
                                  float inv, int32_t zp_off) {
    const __m512  vinv  = _mm512_set1_ps(inv);
    const __m512  vzof  = _mm512_set1_ps((float)zp_off);
    const __m128i vmin8 = _mm_set1_epi8(-127);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512  v = _mm512_fmadd_ps(_mm512_loadu_ps(p + i), vinv, vzof);
        __m512i q = _mm512_cvtps_epi32(v);
        __m128i b = _mm512_cvtsepi32_epi8(q);
        b = _mm_max_epi8(b, vmin8);
        _mm_storeu_si128((__m128i *)(out + i), b);
    }
    if (i < n) {
        __mmask16 m = (__mmask16)((1u << (n - i)) - 1);
        __m512  v = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, p + i), vinv, vzof);
        __m512i q = _mm512_cvtps_epi32(v);
        __m128i b = _mm_max_epi8(_mm512_cvtsepi32_epi8(q), vmin8);
        _mm_mask_storeu_epi8(out + i, m, b);
    }
}
#endif /* x86 */

FeActQuant fe_quantize_activation(const float *A, int total, int8_t *A_q) {
    float vmin = 0.0f, vmax = 0.0f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (fe_qg_x86_avx512) {
        fe_qg_min_max_z(A, total, &vmin, &vmax);
        FeActQuant q = pick_qparams(vmin, vmax);
        fe_qg_quantize_asym_z(A, total, A_q, 1.0f / q.scale, q.zp - 128);
        return q;
    }
#endif
    fe_qg_min_max(A, total, &vmin, &vmax);
    FeActQuant q = pick_qparams(vmin, vmax);
    fe_qg_quantize_asym(A, total, A_q, 1.0f / q.scale, q.zp - 128);
    return q;
}

/* fp16-input quantize. Scans + quantizes directly
 * from the fp16 storage; no fp32 intermediate. Same asymmetric uint8
 * semantics as fe_quantize_activation. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
FeActQuant fe_quantize_activation_fp16(const uint16_t *A_fp16, int total,
                                        int8_t *A_q) {
    float vmin = 0.0f, vmax = 0.0f;
    fe_fp16_min_max(A_fp16, total, &vmin, &vmax);
    FeActQuant q = pick_qparams(vmin, vmax);
    fe_fp16_quantize_asym(A_fp16, total, A_q, 1.0f / q.scale, q.zp - 128);
    return q;
}

/* Read fp32 [K, M] (row-major), write int8 [M, K] (row-major) -- same
 * asymmetric uint8 -> shifted-int8 semantics as fe_quantize_activation,
 * but fused with the transpose so the engine can hand a [F, C]-layout
 * buffer directly to a GEMM that wants [C, F] activations.
 *
 * Implementation: min/max scan is memory-order-invariant (pure reduction
 * over a flat array), so it reuses the contiguous SIMD scan. The quantize
 * loop uses a 4x4 fp32 tile transpose then per-row int8 saturating
 * quantize. Scalar tail handles the M%4 / K%4 remainder.                        */
FeActQuant fe_quantize_activation_transposed(const float *A_KM,
                                              int M, int K,
                                              int8_t *A_q_MK) {
    float vmin = 0.0f, vmax = 0.0f;
    fe_qg_min_max(A_KM, M * K, &vmin, &vmax);
    FeActQuant q = pick_qparams(vmin, vmax);
    const float   inv = 1.0f / q.scale;
    const int32_t zof = q.zp - 128;
    const float   zoff = (float)zof;

    int m = 0;
#if defined(FE_QGEMM_NEON)
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vzof = vdupq_n_f32(zoff);
    const int M4 = M & ~3;
    const int K4 = K & ~3;
    for (; m < M4; m += 4) {
        int k = 0;
        for (; k < K4; k += 4) {
            float32x4_t r0 = vld1q_f32(A_KM + (size_t)(k + 0) * M + m);
            float32x4_t r1 = vld1q_f32(A_KM + (size_t)(k + 1) * M + m);
            float32x4_t r2 = vld1q_f32(A_KM + (size_t)(k + 2) * M + m);
            float32x4_t r3 = vld1q_f32(A_KM + (size_t)(k + 3) * M + m);
            /* 4x4 fp32 transpose using vtrnq + vget_low/high.
             * After this, c[r] = column r of the input tile (= row r of A_q
             * output). */
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);
            float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]),
                                          vget_low_f32(t23.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]),
                                          vget_low_f32(t23.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]),
                                          vget_high_f32(t23.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]),
                                          vget_high_f32(t23.val[1]));
            int32x4_t q0 = vcvtnq_s32_f32(vfmaq_f32(vzof, c0, vinv));
            int32x4_t q1 = vcvtnq_s32_f32(vfmaq_f32(vzof, c1, vinv));
            int32x4_t q2 = vcvtnq_s32_f32(vfmaq_f32(vzof, c2, vinv));
            int32x4_t q3 = vcvtnq_s32_f32(vfmaq_f32(vzof, c3, vinv));
            int16x4_t s0 = vqmovn_s32(q0), s1 = vqmovn_s32(q1);
            int16x4_t s2 = vqmovn_s32(q2), s3 = vqmovn_s32(q3);
            int8x8_t  b01 = vqmovn_s16(vcombine_s16(s0, s1));
            int8x8_t  b23 = vqmovn_s16(vcombine_s16(s2, s3));
            b01 = vmax_s8(b01, vdup_n_s8(-127));
            b23 = vmax_s8(b23, vdup_n_s8(-127));
            uint32_t  w0 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b01), 0);
            uint32_t  w1 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b01), 1);
            uint32_t  w2 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b23), 0);
            uint32_t  w3 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b23), 1);
            memcpy(A_q_MK + (size_t)(m + 0) * K + k, &w0, 4);
            memcpy(A_q_MK + (size_t)(m + 1) * K + k, &w1, 4);
            memcpy(A_q_MK + (size_t)(m + 2) * K + k, &w2, 4);
            memcpy(A_q_MK + (size_t)(m + 3) * K + k, &w3, 4);
        }
        for (; k < K; ++k) {
            for (int r = 0; r < 4; ++r) {
                int qq = (int)lroundf(A_KM[(size_t)k * M + (m + r)] * inv) + zof;
                if (qq < -127) qq = -127;
                if (qq > 127) qq = 127;
                A_q_MK[(size_t)(m + r) * K + k] = (int8_t)qq;
            }
        }
    }
#elif defined(FE_QPOST_AVX2)
    /* x86 128-bit path: 4x4 fp32 transpose via _MM_TRANSPOSE4_PS, then
     * saturating int32 -> int16 -> int8 narrow. The four 32-bit output
     * chunks (one per output row) come out of one packed __m128i; we
     * extract them with srli_si128 shifts (baseline 128-bit only, no
     * dependency on later x86 ISA extensions). qgemm_quant.c builds at
     * baseline so the wider runtime GEMM tiers stay available downstream. */
    const __m128 vinv = _mm_set1_ps(inv);
    const __m128 vzof = _mm_set1_ps(zoff);
    const int M4 = M & ~3;
    const int K4 = K & ~3;
    for (; m < M4; m += 4) {
        int k = 0;
        for (; k < K4; k += 4) {
            __m128 r0 = _mm_loadu_ps(A_KM + (size_t)(k + 0) * M + m);
            __m128 r1 = _mm_loadu_ps(A_KM + (size_t)(k + 1) * M + m);
            __m128 r2 = _mm_loadu_ps(A_KM + (size_t)(k + 2) * M + m);
            __m128 r3 = _mm_loadu_ps(A_KM + (size_t)(k + 3) * M + m);
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);   /* r0..r3 -> 4 output rows */
            __m128i q0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(r0, vinv), vzof));
            __m128i q1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(r1, vinv), vzof));
            __m128i q2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(r2, vinv), vzof));
            __m128i q3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(r3, vinv), vzof));
            __m128i s01 = _mm_packs_epi32(q0, q1);      /* 8 i16 saturating */
            __m128i s23 = _mm_packs_epi32(q2, q3);
            __m128i b   = _mm_packs_epi16(s01, s23);    /* 16 i8 saturating */
            b = _mm_max_epi8(b, _mm_set1_epi8(-127));
            /* b lane layout: [m+0 lanes][m+1 lanes][m+2 lanes][m+3 lanes],
             * each block 4 bytes. Extract via 32-bit shift+cvtsi. */
            int32_t w0 = _mm_cvtsi128_si32(b);
            int32_t w1 = _mm_cvtsi128_si32(_mm_srli_si128(b, 4));
            int32_t w2 = _mm_cvtsi128_si32(_mm_srli_si128(b, 8));
            int32_t w3 = _mm_cvtsi128_si32(_mm_srli_si128(b, 12));
            memcpy(A_q_MK + (size_t)(m + 0) * K + k, &w0, 4);
            memcpy(A_q_MK + (size_t)(m + 1) * K + k, &w1, 4);
            memcpy(A_q_MK + (size_t)(m + 2) * K + k, &w2, 4);
            memcpy(A_q_MK + (size_t)(m + 3) * K + k, &w3, 4);
        }
        for (; k < K; ++k) {
            for (int r = 0; r < 4; ++r) {
                int qq = (int)lroundf(A_KM[(size_t)k * M + (m + r)] * inv) + zof;
                if (qq < -127) qq = -127;
                if (qq > 127) qq = 127;
                A_q_MK[(size_t)(m + r) * K + k] = (int8_t)qq;
            }
        }
    }
#endif
    for (; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            int qq = (int)lroundf(A_KM[(size_t)k * M + m] * inv) + zof;
            if (qq < -127) qq = -127;
            if (qq > 127) qq = 127;
            A_q_MK[(size_t)m * K + k] = (int8_t)qq;
        }
    }
    return q;
}

void fe_quantize_activation_with_scale(const float *A, int total,
                                       int8_t *A_q, float scale,
                                       int32_t zp) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* AVX-512 tier uses the zmm clone (QKV-out re-quant + concat2 per-row
     * land here). Bit-identical to the AVX2 path. */
    if (fe_qg_x86_avx512) {
        fe_qg_quantize_asym_z(A, total, A_q, 1.0f / scale, zp - 128);
        return;
    }
#endif
    fe_qg_quantize_asym(A, total, A_q, 1.0f / scale, zp - 128);
}

/* fp16-input transposed quantize.
 *
 * Reads fp16 [K, M] (column-major view of enc_skip), writes int8 [M, K]
 * with the same asymmetric uint8 semantics as fe_quantize_activation_transposed
 * but with on-the-fly fp16->fp32 cvt during min/max + quantize. No fp32
 * intermediate buffer.
 *
 * SIMD strategy: 4x4 tile (4 K-rows × 4 M-cols), load 4 fp16 lanes per
 * row via VCVTPH2PS/FCVTL, 4x4 fp32 transpose, then per-output-row
 * asymmetric quantize (the same primitive as the fp32 variant — fp16
 * cvt only changes the load side).
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
__attribute__((target("avx2,f16c,fma")))
#endif
FeActQuant fe_quantize_activation_transposed_fp16(const uint16_t *A_KM_fp16,
                                                   int M, int K,
                                                   int8_t *A_q_MK) {
    float vmin = 0.0f, vmax = 0.0f;
    fe_fp16_min_max(A_KM_fp16, M * K, &vmin, &vmax);
    FeActQuant q = pick_qparams(vmin, vmax);
    const float   inv = 1.0f / q.scale;
    const int32_t zof = q.zp - 128;
    const float   zoff = (float)zof;
    (void)inv; (void)zof; (void)zoff;

    /* Dimension contract: M and K must be multiples of 4 (engine sites
     * pass M=FE_C1=96, K=FE_F1=128 — both 4-aligned). Misalignment
     * (m != M || k != K leftover) → abort. */
    int m = 0;
#if defined(FE_QGEMM_NEON)
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vzof = vdupq_n_f32(zoff);
    const int M4 = M & ~3;
    const int K4 = K & ~3;
    for (; m < M4; m += 4) {
        int k = 0;
        for (; k < K4; k += 4) {
            /* Load 4 fp16 lanes per row, cvt to fp32. */
            float32x4_t r0 = fe_fp16_load4(A_KM_fp16 + (size_t)(k + 0) * M + m);
            float32x4_t r1 = fe_fp16_load4(A_KM_fp16 + (size_t)(k + 1) * M + m);
            float32x4_t r2 = fe_fp16_load4(A_KM_fp16 + (size_t)(k + 2) * M + m);
            float32x4_t r3 = fe_fp16_load4(A_KM_fp16 + (size_t)(k + 3) * M + m);
            /* 4x4 fp32 transpose (matches fp32 variant). */
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);
            float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]),
                                          vget_low_f32(t23.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]),
                                          vget_low_f32(t23.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]),
                                          vget_high_f32(t23.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]),
                                          vget_high_f32(t23.val[1]));
            int32x4_t q0 = vcvtnq_s32_f32(vfmaq_f32(vzof, c0, vinv));
            int32x4_t q1 = vcvtnq_s32_f32(vfmaq_f32(vzof, c1, vinv));
            int32x4_t q2 = vcvtnq_s32_f32(vfmaq_f32(vzof, c2, vinv));
            int32x4_t q3 = vcvtnq_s32_f32(vfmaq_f32(vzof, c3, vinv));
            int16x4_t s0 = vqmovn_s32(q0), s1 = vqmovn_s32(q1);
            int16x4_t s2 = vqmovn_s32(q2), s3 = vqmovn_s32(q3);
            int8x8_t  b01 = vqmovn_s16(vcombine_s16(s0, s1));
            int8x8_t  b23 = vqmovn_s16(vcombine_s16(s2, s3));
            b01 = vmax_s8(b01, vdup_n_s8(-127));
            b23 = vmax_s8(b23, vdup_n_s8(-127));
            uint32_t  w0 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b01), 0);
            uint32_t  w1 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b01), 1);
            uint32_t  w2 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b23), 0);
            uint32_t  w3 = (uint32_t)vget_lane_u32(vreinterpret_u32_s8(b23), 1);
            memcpy(A_q_MK + (size_t)(m + 0) * K + k, &w0, 4);
            memcpy(A_q_MK + (size_t)(m + 1) * K + k, &w1, 4);
            memcpy(A_q_MK + (size_t)(m + 2) * K + k, &w2, 4);
            memcpy(A_q_MK + (size_t)(m + 3) * K + k, &w3, 4);
        }
        if (k != K) fe_fp16_alignment_violation();
    }
#elif defined(FE_QPOST_AVX2)
    /* x86: 4x4 fp32 transpose with on-the-fly fp16 load via _mm_cvtph_ps
     * (128-bit form, F16C). Output narrow via _mm_packs_epi32/_mm_packs_epi16. */
    const __m128 vinv = _mm_set1_ps(inv);
    const __m128 vzof = _mm_set1_ps(zoff);
    const int M4 = M & ~3;
    const int K4 = K & ~3;
    for (; m < M4; m += 4) {
        int k = 0;
        for (; k < K4; k += 4) {
            /* Load 4 fp16 (8 bytes) per row, cvt to 4 fp32. */
            __m128 r0 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(A_KM_fp16 + (size_t)(k + 0) * M + m)));
            __m128 r1 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(A_KM_fp16 + (size_t)(k + 1) * M + m)));
            __m128 r2 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(A_KM_fp16 + (size_t)(k + 2) * M + m)));
            __m128 r3 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(A_KM_fp16 + (size_t)(k + 3) * M + m)));
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            __m128i q0 = _mm_cvtps_epi32(_mm_fmadd_ps(r0, vinv, vzof));
            __m128i q1 = _mm_cvtps_epi32(_mm_fmadd_ps(r1, vinv, vzof));
            __m128i q2 = _mm_cvtps_epi32(_mm_fmadd_ps(r2, vinv, vzof));
            __m128i q3 = _mm_cvtps_epi32(_mm_fmadd_ps(r3, vinv, vzof));
            __m128i s01 = _mm_packs_epi32(q0, q1);
            __m128i s23 = _mm_packs_epi32(q2, q3);
            __m128i b   = _mm_packs_epi16(s01, s23);
            b = _mm_max_epi8(b, _mm_set1_epi8(-127));
            int32_t w0 = _mm_cvtsi128_si32(b);
            int32_t w1 = _mm_cvtsi128_si32(_mm_srli_si128(b, 4));
            int32_t w2 = _mm_cvtsi128_si32(_mm_srli_si128(b, 8));
            int32_t w3 = _mm_cvtsi128_si32(_mm_srli_si128(b, 12));
            memcpy(A_q_MK + (size_t)(m + 0) * K + k, &w0, 4);
            memcpy(A_q_MK + (size_t)(m + 1) * K + k, &w1, 4);
            memcpy(A_q_MK + (size_t)(m + 2) * K + k, &w2, 4);
            memcpy(A_q_MK + (size_t)(m + 3) * K + k, &w3, 4);
        }
        if (k != K) fe_fp16_alignment_violation();
    }
#endif
    if (m != M) fe_fp16_alignment_violation();
    return q;
}

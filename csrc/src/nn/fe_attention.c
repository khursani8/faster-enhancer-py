/*
 * Multi-head self-attention over the frequency axis. Full int8 W8A8 for
 * QKV proj, Q*K^T and S*V; softmax in fp32. Per-tensor qkv_scale shared
 * across Q/K/V; s_scale = 1/127 quantises softmax. K/V for all NH heads
 * are packed in a single pass over qkv_q.
 */
#include "fe_internal.h"
#include "fe_sgemm.h"
#include "fe_qgemm.h"
#include "fe_simd.h"
#include "fe_profile.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_dispatch.h"
#include "qgemm/arch_kernels.h"
#include <math.h>
#include <string.h>

#define FE_QGEMM_NR_LOCAL 8

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/*
 * Single-pass Q/K/V extract+pack for all NH heads. Each qkv_q row read
 * once. Q is contiguous [F2, HD]; K is DOTPROD-packed
 * [block_n=F2/NR][g=HD/4][NR][4]; V is transposed into
 * [block_n=HD/NR][g=F2/4][NR][4]. I8MM repack folds in at the bottom.
 */
static void pack_all_heads_int8(
        const int8_t *qkv_q,
        int F2, int NH, int HD, int C3,
        int8_t *Qq_all,   // [NH, F2, HD_PAD_K]
        int8_t *Kpq_all,  // [NH, F2/NR * HD_PAD_K/4 * NR * 4]
        int8_t *Vpq_all)  // [NH, HD_PAD_V/NR * F2/4 * NR * 4]
{
    const int NR = FE_QGEMM_NR_LOCAL;
    const int HD_K = FE_HEAD_DIM_PAD_K;
    const int HD_V = FE_HEAD_DIM_PAD_V;
    const int K_blocks_n = F2 / NR;
    const int K_g_count  = HD_K / 4;
    const int V_blocks_n = HD_V / NR;

    const size_t K_head_stride = (size_t)K_blocks_n * K_g_count * NR * 4;
    const size_t V_head_stride = (size_t)V_blocks_n * (F2 / 4) * NR * 4;
    const size_t Q_head_stride = (size_t)F2 * HD_K;

    /* Zero V pack regions; Q/K packs are fully overwritten below. */
    memset(Vpq_all, 0, (size_t)NH * V_head_stride);

    /* Batch 4 m-rows: v_klane = m & 3 cycles 0..3 so the V scatter for
     * a fixed (v_g, v_bn, v_lane) lands in 4 contiguous bytes -- the
     * shape vst4_s8 produces. Medium F2 is a multiple of 4; scalar
     * tail kept for safety. */
    int m_block_4 = F2 & ~3;
    int m_blocks  = m_block_4 / 4;
    for (int mb = 0; mb < m_blocks; ++mb) {
        int m_base   = mb * 4;
        int v_g      = m_base / 4;
        int bn       = m_base / NR;
        int lane_lo  = m_base % NR;       /* 0 or 4 */

        for (int h = 0; h < NH; ++h) {
            const int8_t *row0 = qkv_q + (size_t)(m_base + 0) * C3 + h * 3 * HD;
            const int8_t *row1 = qkv_q + (size_t)(m_base + 1) * C3 + h * 3 * HD;
            const int8_t *row2 = qkv_q + (size_t)(m_base + 2) * C3 + h * 3 * HD;
            const int8_t *row3 = qkv_q + (size_t)(m_base + 3) * C3 + h * 3 * HD;
            const int8_t *q0 = row0, *q1 = row1, *q2 = row2, *q3 = row3;
            const int8_t *k0 = row0 + HD, *k1 = row1 + HD;
            const int8_t *k2 = row2 + HD, *k3 = row3 + HD;
            const int8_t *v0 = row0 + 2*HD, *v1 = row1 + 2*HD;
            const int8_t *v2 = row2 + 2*HD, *v3 = row3 + 2*HD;

            /* Q: 4 contiguous rows, each [HD bytes data + (HD_K - HD) zeros]. */
            int8_t *q_dst0 = Qq_all + (size_t)h * Q_head_stride + (size_t)(m_base + 0) * HD_K;
            memcpy(q_dst0,         q0, (size_t)HD);
            memcpy(q_dst0 + HD_K,  q1, (size_t)HD);
            memcpy(q_dst0 + 2*HD_K,q2, (size_t)HD);
            memcpy(q_dst0 + 3*HD_K,q3, (size_t)HD);
            for (int r = 0; r < 4; ++r)
                for (int t = HD; t < HD_K; ++t) q_dst0[r * HD_K + t] = 0;

            /* K: for each g, 4 m's write 16 contiguous bytes; partial-g
             * tail zero-pads past HD. */
            int8_t *kp_blk = Kpq_all + (size_t)h * K_head_stride
                                     + (size_t)bn * K_g_count * NR * 4;
            int g_full = HD / 4;          /* full 4-byte k_lane groups */
            for (int g = 0; g < g_full; ++g) {
                int base = g * 4;
                uint32_t w0, w1, w2, w3;
                memcpy(&w0, k0 + base, 4); memcpy(&w1, k1 + base, 4);
                memcpy(&w2, k2 + base, 4); memcpy(&w3, k3 + base, 4);
#if defined(__ARM_NEON) || defined(__aarch64__)
                int32x4_t vw = {(int32_t)w0, (int32_t)w1,
                                (int32_t)w2, (int32_t)w3};
                vst1q_s8(kp_blk + g * NR * 4 + lane_lo * 4,
                         vreinterpretq_s8_s32(vw));
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                __m128i vw = _mm_set_epi32((int)w3, (int)w2, (int)w1, (int)w0);
                _mm_storeu_si128((__m128i *)(kp_blk + g * NR * 4 + lane_lo * 4),
                                 vw);
#endif
            }
            for (int g = g_full; g < K_g_count; ++g) {
                /* Partial g: pad past HD with zero. */
                int base = g * 4;
                int8_t *kp = kp_blk + g * NR * 4 + lane_lo * 4;
                const int8_t *ks[4] = { k0, k1, k2, k3 };
                for (int r = 0; r < 4; ++r) {
                    for (int kl = 0; kl < 4; ++kl) {
                        kp[r * 4 + kl] = (base + kl < HD) ? ks[r][base + kl] : 0;
                    }
                }
            }

            /* V: full 8-lane (v_bn) blocks via vst4_s8 -- 4 m's interleaved
             * into 32 contiguous bytes. */
            int8_t *vp_head = Vpq_all + (size_t)h * V_head_stride;
            int v_bn_full = HD / 8;
            for (int v_bn = 0; v_bn < v_bn_full; ++v_bn) {
                int8_t *vp_blk = vp_head + (size_t)v_bn * (F2 / 4) * NR * 4
                                         + (size_t)v_g * NR * 4;
#if defined(__ARM_NEON) || defined(__aarch64__)
                int8x8x4_t quad = {{
                    vld1_s8(v0 + v_bn * 8),
                    vld1_s8(v1 + v_bn * 8),
                    vld1_s8(v2 + v_bn * 8),
                    vld1_s8(v3 + v_bn * 8)
                }};
                vst4_s8(vp_blk, quad);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                /* 128-bit 4-stream byte interleave: matches vst4_s8 layout. */
                __m128i lv0 = _mm_loadl_epi64((const __m128i *)(v0 + v_bn * 8));
                __m128i lv1 = _mm_loadl_epi64((const __m128i *)(v1 + v_bn * 8));
                __m128i lv2 = _mm_loadl_epi64((const __m128i *)(v2 + v_bn * 8));
                __m128i lv3 = _mm_loadl_epi64((const __m128i *)(v3 + v_bn * 8));
                __m128i ab  = _mm_unpacklo_epi8(lv0, lv1);  /* (a,b) pairs */
                __m128i cd  = _mm_unpacklo_epi8(lv2, lv3);  /* (c,d) pairs */
                __m128i lo  = _mm_unpacklo_epi16(ab, cd);
                __m128i hi  = _mm_unpackhi_epi16(ab, cd);
                _mm_storeu_si128((__m128i *)(vp_blk +  0), lo);
                _mm_storeu_si128((__m128i *)(vp_blk + 16), hi);
#endif
            }
            /* Partial last v_bn (HD % 8 valid lanes). */
            int v_bn_rem = HD - v_bn_full * 8;
            if (v_bn_rem > 0) {
                int v_bn = v_bn_full;
                int8_t *vp_blk = vp_head + (size_t)v_bn * (F2 / 4) * NR * 4
                                         + (size_t)v_g * NR * 4;
                for (int n = 0; n < v_bn_rem; ++n) {
                    int idx = v_bn * 8 + n;
                    vp_blk[n * 4 + 0] = v0[idx];
                    vp_blk[n * 4 + 1] = v1[idx];
                    vp_blk[n * 4 + 2] = v2[idx];
                    vp_blk[n * 4 + 3] = v3[idx];
                }
            }
        }
    }

    /* Scalar tail for F2 % 4 != 0. */
    for (int m = m_block_4; m < F2; ++m) {
        const int8_t *row = qkv_q + (size_t)m * C3;
        const int bn      = m / NR;
        const int lane    = m % NR;
        const int v_g     = m / 4;
        const int v_klane = m % 4;
        for (int h = 0; h < NH; ++h) {
            const int8_t *head = row + h * 3 * HD;
            const int8_t *q_src = head;
            const int8_t *k_src = head + HD;
            const int8_t *v_src = head + 2 * HD;
            int8_t *q_dst = Qq_all + (size_t)h * Q_head_stride
                                   + (size_t)m * HD_K;
            memcpy(q_dst, q_src, (size_t)HD);
            for (int t = HD; t < HD_K; ++t) q_dst[t] = 0;
            int8_t *kp_head = Kpq_all + (size_t)h * K_head_stride;
            int8_t *kp_blk  = kp_head + (size_t)bn * K_g_count * NR * 4;
            for (int g = 0; g < K_g_count; ++g) {
                int8_t *kp = kp_blk + (size_t)g * NR * 4 + lane * 4;
                const int base = g * 4;
                kp[0] = (base + 0 < HD) ? k_src[base + 0] : 0;
                kp[1] = (base + 1 < HD) ? k_src[base + 1] : 0;
                kp[2] = (base + 2 < HD) ? k_src[base + 2] : 0;
                kp[3] = (base + 3 < HD) ? k_src[base + 3] : 0;
            }
            int8_t *vp_head = Vpq_all + (size_t)h * V_head_stride;
            for (int n = 0; n < HD; ++n) {
                int v_bn   = n >> 3;
                int v_lane = n & 7;
                int8_t *vp_blk = vp_head + (size_t)v_bn * (F2 / 4) * NR * 4;
                int8_t *vp = vp_blk + (size_t)v_g * NR * 4 + v_lane * 4 + v_klane;
                *vp = v_src[n];
            }
        }
    }

    /* I8MM tier: in-place repack each N-block from DOTPROD layout.
     * Other tiers consume DOTPROD directly. */
    if (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM) {
        int8_t scratch[FE_QGEMM_MAX_K * 8];
        const int K_KdimK = HD;
        const int V_KdimK = F2;
        for (int h = 0; h < NH; ++h) {
            int8_t *kp_head = Kpq_all + (size_t)h * K_head_stride;
            if ((K_KdimK & 7) == 0) {
                for (int bn = 0; bn < K_blocks_n; ++bn) {
                    int8_t *blk = kp_head + (size_t)bn * K_g_count * NR * 4;
                    fe_qgemm_i8mm_repack_block(blk, K_KdimK, scratch);
                    memcpy(blk, scratch, (size_t)(K_KdimK / 8) * 64);
                }
            }
            int8_t *vp_head = Vpq_all + (size_t)h * V_head_stride;
            if ((V_KdimK & 7) == 0) {
                for (int bn = 0; bn < V_blocks_n; ++bn) {
                    int8_t *blk = vp_head + (size_t)bn * (F2 / 4) * NR * 4;
                    fe_qgemm_i8mm_repack_block(blk, V_KdimK, scratch);
                    memcpy(blk, scratch, (size_t)(V_KdimK / 8) * 64);
                }
            }
        }
    }
}

void fe_mhsa(FeAttention *a, FeLinear *fc,
             const float *in, float *out,
             float *qkv_buf, float *score_buf, float *attn_buf,
             int8_t *qkv_q, int8_t *Qq, int8_t *Kpq, int8_t *Vpq,
             int8_t *scoresq,
             int freq,
             int8_t *aq, int32_t *c32) {
    const int C  = FE_C2;
    const int NH = FE_NUM_HEADS;
    const int HD = FE_HEAD_DIM;
    const int C3 = 3 * C;
    const float scale = 1.0f / sqrtf((float)HD);

    FeActQuant qkv_q_params;
    FE_TIME_BEGIN("attn_1_qkv_proj");
    qkv_q_params = fe_qgemm_packed_calib_to_int8out(freq, C3, C, in,
                                                 a->qkv.weight_q, a->qkv.scales_w,
                                                 a->qkv.row_sums,
                                                 a->qkv.bias,
                                                 qkv_q, C3,
                                                 qkv_buf, C3,
                                                 aq, c32,
                                                 &a->qkv.act, &a->qkv.act_out);
    FE_TIME_END();
    const float qkv_scale = qkv_q_params.scale;

    const int HD_K = FE_HEAD_DIM_PAD_K;
    const int HD_V = FE_HEAD_DIM_PAD_V;

    FE_TIME("attn_2_pack", pack_all_heads_int8(qkv_q, freq, NH, HD, C3, Qq, Kpq, Vpq));

    const size_t Q_head_stride = (size_t)freq * HD_K;
    const size_t K_head_stride = (size_t)(freq / FE_QGEMM_NR_LOCAL)
                               * (HD_K / 4) * FE_QGEMM_NR_LOCAL * 4;
    const size_t V_head_stride = (size_t)(HD_V / FE_QGEMM_NR_LOCAL)
                               * (freq / 4) * FE_QGEMM_NR_LOCAL * 4;
    const float s_scale = 1.0f / 127.0f;

    /* Q@K^T has a uniform per-output scale (qkv_scale * 1/sqrt(HD) for
     * all n), so the combined dequant factor is a single scalar:
     * qkv_scale^2 / sqrt(HD). gemm_int32 writes c32 directly; softmax
     * dequants inline in pass-1. */
    const float qk_combined_scale = qkv_scale * qkv_scale * scale;

    for (int h = 0; h < NH; ++h) {
        const int8_t *Qq_h  = Qq  + (size_t)h * Q_head_stride;
        const int8_t *Kpq_h = Kpq + (size_t)h * K_head_stride;
        const int8_t *Vpq_h = Vpq + (size_t)h * V_head_stride;

        /* QKV proj + softmax both emit symmetric int8 (zp=128), so the
         * (128-zp) correction is 0 (NULL row_sums). Specialised K=20
         * kernels apply where K%8 != 0 forces I8MM -> DOTPROD. */
        FE_TIME_BEGIN("attn_3_qk");
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
        if (HD_K == 20 &&
            (fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_DOTPROD
             || fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_I8MM)) {
            qgemm_dotprod_int32_k20(freq, freq, Qq_h, Kpq_h, c32, freq);
        } else if (HD_K == 20 && fe_qgemm_ops.tier == FE_QGEMM_TIER_ARM_NEON) {
            qgemm_neon_int32_k20(freq, freq, Qq_h, Kpq_h, c32, freq);
        } else
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        if (HD_K == 20 && fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX512_VNNI) {
            qgemm_avx512vnni_int32_k20(freq, freq, Qq_h, Kpq_h, c32, freq);
        } else if (HD_K == 20 && fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX_VNNI) {
            qgemm_avxvnni_int32_k20(freq, freq, Qq_h, Kpq_h, c32, freq);
        } else if (HD_K == 20 && fe_qgemm_ops.tier == FE_QGEMM_TIER_X86_AVX2) {
            qgemm_avx2_int32_k20(freq, freq, Qq_h, Kpq_h, c32, freq);
        } else
#endif
        {
            fe_qgemm_ops.gemm_int32(freq, freq, HD_K, Qq_h, Kpq_h,
                                    c32, freq);
        }
        FE_TIME_END();
        FE_TIME("14a_softmax_quant",
                fe_softmax_rows_quant_from_int32(c32, freq, freq,
                                                  qk_combined_scale,
                                                  score_buf,
                                                  scoresq, 1.0f / s_scale));
        /* S@V mirrors Q@K: gemm_int32 + uniform combined scale
         * (s_scale * qkv_scale = qkv_scale/127). Dequant folds into the
         * strided HD_V -> HD copy that fills attn_buf per head. */
        FE_TIME("attn_5_sv",
                fe_qgemm_ops.gemm_int32(freq, HD_V, freq, scoresq, Vpq_h,
                                        c32, HD_V));
        const float sv_combined = s_scale * qkv_scale;
        for (int r = 0; r < freq; ++r) {
            const int32_t *src = c32 + (size_t)r * HD_V;
            float         *dst = attn_buf + (size_t)r * C + h * HD;
            int n = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
            const __m512 vcs16 = _mm512_set1_ps(sv_combined);
            for (; n + 15 < HD; n += 16) {
                __m512i vi = _mm512_loadu_si512((const __m512i *)(src + n));
                _mm512_storeu_ps(dst + n,
                    _mm512_mul_ps(_mm512_cvtepi32_ps(vi), vcs16));
            }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
            const __m256 vcs8 = _mm256_set1_ps(sv_combined);
            for (; n + 7 < HD; n += 8) {
                __m256i vi = _mm256_loadu_si256((const __m256i *)(src + n));
                _mm256_storeu_ps(dst + n,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(vi), vcs8));
            }
#endif
#if defined(__ARM_NEON)
            const float32x4_t vcs = vdupq_n_f32(sv_combined);
            for (; n + 3 < HD; n += 4) {
                int32x4_t   vi = vld1q_s32(src + n);
                float32x4_t vf = vmulq_f32(vcvtq_f32_s32(vi), vcs);
                vst1q_f32(dst + n, vf);
            }
            /* 2-wide NEON tail (handles HD % 4 == 2). */
            if (n + 1 < HD) {
                int32x2_t   vi = vld1_s32(src + n);
                float32x2_t vf = vmul_f32(vcvt_f32_s32(vi),
                                          vget_low_f32(vcs));
                vst1_f32(dst + n, vf);
                n += 2;
            }
#endif
            for (; n < HD; ++n)
                dst[n] = (float)src[n] * sv_combined;
        }
    }

    /* attn_6_fc + resB fused: out += bias + scale*c32. Caller passes
     * the residual base (rf_b) as out. fp32_scratch unreachable on ARM. */
    FE_TIME("attn_6_fc",
            fe_qgemm_packed_calib_acc(freq, C, C, attn_buf,
                                       fc->weight_q, fc->scales_w, fc->row_sums,
                                       fc->bias, out, C,
                                       aq, c32, NULL, &fc->act));
}

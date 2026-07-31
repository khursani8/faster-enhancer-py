/*
 * Per-frame inference graph: STFT -> compress -> encoder -> rnnformer
 * -> decoder -> mask -> iSTFT. Heavy matmuls route through fe_qgemm
 * with runtime SIMD dispatch; k=3 conv uses Winograd F(2,3).
 */
#include "fe_internal.h"
#include "fe_sgemm.h"
#include "fe_qgemm.h"
/* Defines FE_QGEMM_HAVE_AVX2 / _AVXVNNI / _AVX512_VNNI. Without it the
 * wide complex mask-multiply below is preprocessed out on x86 and the
 * 128-bit remainder loop silently does the whole 512-bin pass. */
#include "qgemm/qgemm_arch.h"
#include "fe_simd.h"
#include "fe_fp16.h"
#include "fe_profile.h"
#include <stdlib.h>
#include <string.h>

FeState *fe_state_create(void) {
    FeState *s = (FeState *)calloc(1, sizeof(FeState));
    if (!s) return NULL;
    fe_fft_init();
    fe_stft_init(s);
    /* MHSA and Winograd scratch share the same union (never live at once). */
    fe_winograd_set_scratch(s->wino_U, s->wino_dq);
    return s;
}

void fe_state_destroy(FeState *s) {
    if (s) free(s);
}

/*
 * Freq-axis Linear with input transpose folded into the int8 quantize.
 * In: [F_in, C], out: [C, F_out]. Underlying GEMM: M=C, K=F_in, N=F_out.
 */
static inline void apply_freq_linear_FCin(FeLinear *lin,
                                          const float *in_FCin, float *out,
                                          int channels, int f_in, int f_out,
                                          int8_t *aq, int32_t *c32) {
    fe_qgemm_packed_calib_transposed_in(
        channels, f_out, f_in, in_FCin,
        lin->weight_q, lin->scales_w, lin->row_sums,
        lin->bias, out, f_out,
        aq, c32, &lin->act);
}

/*
 * Per-layer profile bucket names. fe_profile_record matches by pointer
 * identity. Arrays compile out when FE_ENABLE_PROFILE is off.
 */
#define UNUSED __attribute__((unused))
static const char *NM_ENC_K3 [FE_ENC_BLOCKS] UNUSED = {
    "enc_b0_k3", "enc_b1_k3", "enc_b2_k3" };
static const char *NM_DEC_1X1[FE_DEC_BLOCKS] UNUSED = {
    "dec_b0_1x1", "dec_b1_1x1", "dec_b2_1x1" };
static const char *NM_DEC_K3 [FE_DEC_BLOCKS] UNUSED = {
    "dec_b0_k3", "dec_b1_k3", "dec_b2_k3" };
static const char *NM_RF_GRU [FE_RF_BLOCKS]  UNUSED = {
    "rf_b0_gru", "rf_b1_gru", "rf_b2_gru", "rf_b3_gru" };
static const char *NM_RF_FC  [FE_RF_BLOCKS]  UNUSED = {
    "rf_b0_fc",  "rf_b1_fc",  "rf_b2_fc",  "rf_b3_fc"  };
static const char *NM_RF_ATTN[FE_RF_BLOCKS]  UNUSED = {
    "rf_b0_attn","rf_b1_attn","rf_b2_attn","rf_b3_attn" };

void fe_process_frame(FeState *s, FeWeights *w,
                      const float *in, float *out) {
    /* 1) STFT -> compressed spectrum in split [re | im] layout. */
    FE_TIME("01_stft", fe_stft(s, in));

    /* Working buffers buf_a/buf_b/enc_skip/rf_* use [F, C] layout. */

    /* 2) Encoder pre-net (strided conv) -- fp32 sgemm; K=2*Ci=16 too small
     * for int8 to amortize. */
    FE_TIME("03_enc_pre_strided",
            fe_strided_conv1d(&w->enc_pre, s->spec_cf, s->buf_a,
                              FE_FREQ_BINS, FE_STRIDE));

    /* 3) Encoder blocks (Conv1d k=3 + SiLU) on ping-pong buf_a/buf_b.
     * enc_skip is stored IEEE fp16 and writes happen kernel-side.
     * fe_silu_skip_fp16 fuses SiLU + fp16 store in one pass (vs. a separate
     * fp32 SiLU pass followed by an fp16 pack). Winograd epilogue
     * variants similarly store fp16 directly via VCVTPS2PH / FCVTN. */
    {
        FE_TIME("04_enc_pre_silu",
                fe_silu_skip_fp16(s->buf_a, FE_C1 * FE_F1, s->enc_skip[0]));

        float *cur = s->buf_a;
        float *nxt = s->buf_b;
        for (int i = 0; i < FE_ENC_BLOCKS; ++i) {
            FE_TIME(NM_ENC_K3[i],
                    fe_conv1d_k3_buf_silu_skip_fp16(&w->enc[i], cur, nxt,
                                                     s->enc_skip[i + 1], FE_F1,
                                                     s->qgemm_aq, s->qgemm_c32));
            float *t = cur; cur = nxt; nxt = t;
        }
        (void)cur;
    }

    /* 4) RNNFormer pre-net: Linear(F1->F2) + Conv1d 1x1 (C1->C2). Both
     * transposes fold into int8 quantize passes (transposed-in trick on
     * the Linear input and the conv K axis), so no rf_lin_TT scratch.
     * transposed_in is fp16-direct: the kernel reads fp16 with
     * on-the-fly cvt during min/max + quantize, no unpack pass. */
    FE_TIME("07_rf_pre_lin",
            fe_qgemm_packed_calib_transposed_in_fp16(
                FE_C1, FE_F2, FE_F1, s->enc_skip[FE_ENC_BLOCKS],
                w->rf_pre_lin.weight_q, w->rf_pre_lin.scales_w,
                w->rf_pre_lin.row_sums, w->rf_pre_lin.bias,
                s->rf_lin_T, FE_F2,
                s->qgemm_aq, s->qgemm_c32, &w->rf_pre_lin.act));
    FE_TIME("08_rf_pre_conv",
            fe_conv1d_k1_FCin(&w->rf_pre_conv, s->rf_lin_T, s->rf_b, FE_F2,
                              s->qgemm_aq, s->qgemm_c32));

    /* 5) RNNFormer blocks (GRU + FC + residual + attention + residual).
     * All SIMD tiers' gru_full_fused_fp16inout dual-stores new state to
     * BOTH s->gru_h[blk] (fp16, VCVTPS2PH / FCVTN) AND gru_h_fp32 stack
     * scratch (rnn_fc consumes the scratch). Engine unpack + pack passes
     * both eliminated. */
    float gru_h_fp32[FE_F2 * FE_C2];
    for (int blk = 0; blk < FE_RF_BLOCKS; ++blk) {
        FeRNNFormerBlock *rb = &w->rf[blk];

        FE_TIME(NM_RF_GRU[blk],
                fe_gru_step_fp16h(&rb->gru, s->rf_b,
                                   s->gru_h[blk], gru_h_fp32, FE_F2,
                                   s->qgemm_aq, s->qgemm_c32));

        /* rnn_fc + resA fused: rf_b += bias + scale*c32 in the kernel
         * epilogue. fp32_scratch unreachable on ARM tiers (NULL). */
        FE_TIME(NM_RF_FC[blk],
                fe_qgemm_packed_calib_acc(FE_F2, FE_C2, FE_C2, gru_h_fp32,
                                           rb->rnn_fc.weight_q, rb->rnn_fc.scales_w,
                                           rb->rnn_fc.row_sums,
                                           rb->rnn_fc.bias,
                                           s->rf_b, FE_C2,
                                           s->qgemm_aq, s->qgemm_c32,
                                           NULL,
                                           &rb->rnn_fc.act));

        if (rb->has_pe && rb->pe) {
            FE_TIME("rf_b0_pe", fe_vec_add(s->rf_b, rb->pe, FE_F2 * FE_C2));
        }

        /* in==out aliasing is safe: QKV proj reads rf_b first, attn_6_fc
         * accumulates back into rf_b last (acc variant absorbs resB). */
        FE_TIME(NM_RF_ATTN[blk],
                fe_mhsa(&rb->attn, &rb->attn_fc,
                        s->rf_b, s->rf_b,
                        s->attn_qkv, s->attn_scores, s->attn_out,
                        s->qkv_q, s->attn_Qq, s->attn_Kpq, s->attn_Vpq,
                        s->attn_scoresq, FE_F2,
                        s->qgemm_aq, s->qgemm_c32));
    }

    /* 6) RNNFormer post-net (both transposes fused, same trick as pre-net). */
    FE_TIME("17_rf_post_lin",
            apply_freq_linear_FCin(&w->rf_post_lin, s->rf_b, s->rf_lin_T,
                                   FE_C2, FE_F2, FE_F1,
                                   s->qgemm_aq, s->qgemm_c32));
    FE_TIME("18_rf_post_conv",
            fe_conv1d_k1_FCin(&w->rf_post_conv, s->rf_lin_T, s->buf_a, FE_F1,
                              s->qgemm_aq, s->qgemm_c32));

    /* 7) Decoder blocks: concat[buf_a | enc_skip] -> conv k=1 + SiLU ->
     * conv k=3 + SiLU. The concat2_fp16b kernel consumes the fp16 enc_skip
     * directly — fp16 load + cvt + asym quantize inline, no fp32 unpack
     * pass. */
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) {
        int skip_idx = FE_ENC_BLOCKS - i;
        /* Prefetch enc_skip[skip_idx] -- cold since enc finished. */
        __builtin_prefetch(s->enc_skip[skip_idx] +  0, 0, 1);
        __builtin_prefetch(s->enc_skip[skip_idx] + 32, 0, 1);
        __builtin_prefetch(s->enc_skip[skip_idx] + 64, 0, 1);
        __builtin_prefetch(s->enc_skip[skip_idx] + 96, 0, 1);
        FE_TIME(NM_DEC_1X1[i],
                fe_conv1d_k1_silu_concat2_fp16b(&w->dec_1x1[i],
                                                 s->buf_a, s->enc_skip[skip_idx],
                                                 s->buf_b, FE_F1,
                                                 s->qgemm_aq, s->qgemm_c32));
        FE_TIME(NM_DEC_K3[i],
                fe_conv1d_k3_buf_silu(&w->dec_3x3[i], s->buf_b, s->buf_a,
                                      FE_F1,
                                      s->qgemm_aq, s->qgemm_c32));
    }

    /* 8) Decoder post-net (enc_skip[0] from enc_pre via fp16 storage).
     * Same fp16-direct concat2 fusion as above. */
    __builtin_prefetch(s->enc_skip[0] +  0, 0, 1);
    __builtin_prefetch(s->enc_skip[0] + 32, 0, 1);
    __builtin_prefetch(s->enc_skip[0] + 64, 0, 1);
    __builtin_prefetch(s->enc_skip[0] + 96, 0, 1);
    FE_TIME("23_dec_post_1x1_silu",
            fe_conv1d_k1_silu_concat2_fp16b(&w->dec_post_1x1,
                                             s->buf_a, s->enc_skip[0],
                                             s->buf_b, FE_F1,
                                             s->qgemm_aq, s->qgemm_c32));
    FE_TIME("25_dec_post_convT",
            fe_conv_transpose1d(&w->dec_post_up, s->buf_b, s->mask_cf, FE_F1));

    /* 9) Complex mask multiply (split layout).
     *    out_re = sr*mr - si*mi ; out_im = sr*mi + si*mr */
    FE_TIME_BEGIN("26_mask_multiply");
    {
        const float *sr_p = s->spec_cf;
        const float *si_p = s->spec_cf + FE_FREQ_BINS;
        const float *mr   = s->mask_cf;
        const float *mi   = s->mask_cf + FE_FREQ_BINS;
        float       *o_re = s->spec_out;
        float       *o_im = s->spec_out + FE_FREQ_BINS;
        int f = 0;
#if defined(FE_QGEMM_HAVE_AVX512_VNNI)
        for (; f + 15 < FE_FREQ_BINS; f += 16) {
            __m512 sr  = _mm512_loadu_ps(sr_p + f);
            __m512 si  = _mm512_loadu_ps(si_p + f);
            __m512 vmr = _mm512_loadu_ps(mr + f);
            __m512 vmi = _mm512_loadu_ps(mi + f);
            _mm512_storeu_ps(o_re + f,
                _mm512_fmsub_ps(sr, vmr, _mm512_mul_ps(si, vmi)));
            _mm512_storeu_ps(o_im + f,
                _mm512_fmadd_ps(sr, vmi, _mm512_mul_ps(si, vmr)));
        }
#elif defined(FE_QGEMM_HAVE_AVX2) || defined(FE_QGEMM_HAVE_AVXVNNI)
        for (; f + 7 < FE_FREQ_BINS; f += 8) {
            __m256 sr  = _mm256_loadu_ps(sr_p + f);
            __m256 si  = _mm256_loadu_ps(si_p + f);
            __m256 vmr = _mm256_loadu_ps(mr + f);
            __m256 vmi = _mm256_loadu_ps(mi + f);
            _mm256_storeu_ps(o_re + f,
                _mm256_fmsub_ps(sr, vmr, _mm256_mul_ps(si, vmi)));
            _mm256_storeu_ps(o_im + f,
                _mm256_fmadd_ps(sr, vmi, _mm256_mul_ps(si, vmr)));
        }
#endif
        for (; f + 3 < FE_FREQ_BINS; f += 4) {
            fe_f32x4 sr  = fe_load(sr_p + f);
            fe_f32x4 si  = fe_load(si_p + f);
            fe_f32x4 vmr = fe_load(mr + f);
            fe_f32x4 vmi = fe_load(mi + f);
            fe_store(o_re + f, fe_sub(fe_mul(sr, vmr), fe_mul(si, vmi)));
            fe_store(o_im + f, fe_fma(sr, vmi, fe_mul(si, vmr)));
        }
        for (; f < FE_FREQ_BINS; ++f) {
            o_re[f] = sr_p[f] * mr[f] - si_p[f] * mi[f];
            o_im[f] = sr_p[f] * mi[f] + si_p[f] * mr[f];
        }
    }
    FE_TIME_END();

    /* 10) iSTFT. */
    FE_TIME("27_istft", fe_istft(s, out));
    (void)out;
}

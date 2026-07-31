// Weight loader. Production accepts exactly one on-disk format:
//   Int8 packed (.q8 blob from tools/quantize_bin.py): magic "FM_W8_03"
//   header, payload is already in DOTPROD-packed layout with per-row scales
//   interleaved. GRU W_ih/W_hh are also stored in prepacked int8 form.
//
// After fe_qgemm_init has resolved the runtime tier, the engine calls
// fe_weights_finalize_for_tier() to do the in-place DOTPROD->I8MM repack
// (no-op for other tiers).
//
// The blob passed to fe_load_weights must outlive use -- biases, PE, and
// fp32-only weights inside q8 (enc_pre, dec_post_up) are zero-copy pointers.
#include "fe_weights.h"
#include "fe_internal.h"
#include "fe_sgemm.h"
#include "fe_qgemm.h"
#include "qgemm/qgemm_arch.h"
#include "qgemm/qgemm_dispatch.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Layer enumeration -- gather pointers to every layer of a given kind in the
 * canonical pack order, returning the count. Shared by the q8 loader and
 * fe_weights_finalize_for_tier (was triplicated;
 * adding a layer now means editing one place). Pure pointer gathering at load
 * time. Caller sizes the arrays: linears[FE_RF_BLOCKS*3 + 2],
 * convs[2 + FE_ENC_BLOCKS + FE_DEC_BLOCKS*2 + 2], k3[FE_ENC_BLOCKS+FE_DEC_BLOCKS]. */
static int collect_linears(FeWeights *w, FeLinear **out) {
    int n = 0;
    for (int blk = 0; blk < FE_RF_BLOCKS; ++blk) {
        out[n++] = &w->rf[blk].rnn_fc;
        out[n++] = &w->rf[blk].attn_fc;
        out[n++] = &w->rf[blk].attn.qkv;
    }
    out[n++] = &w->rf_pre_lin;
    out[n++] = &w->rf_post_lin;
    return n;
}

static int collect_convs(FeWeights *w, FeConv1d **out) {
    int n = 0;
    out[n++] = &w->enc_pre;        /* k=2 after reshape, Ci=stride*2=8 */
    for (int i = 0; i < FE_ENC_BLOCKS; ++i) out[n++] = &w->enc[i];
    out[n++] = &w->rf_pre_conv;
    out[n++] = &w->rf_post_conv;
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) out[n++] = &w->dec_1x1[i];
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) out[n++] = &w->dec_3x3[i];
    out[n++] = &w->dec_post_1x1;
    return n;
}

static int collect_k3_layers(FeWeights *w, FeConv1d **out) {
    int n = 0;
    for (int i = 0; i < FE_ENC_BLOCKS; ++i) out[n++] = &w->enc[i];
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) out[n++] = &w->dec_3x3[i];
    return n;
}

/* ------------------------------------------------------------------ *
 *  Int8 format loader.                                                 *
 *  File layout (matches tools/quantize_bin.py):                        *
 *    [magic "FM_W8_03"] [u32 version=1]                                *
 *    For each layer in fe_load_weights() iteration order:              *
 *      - fp32-only layer (enc_pre, dec_post_up):  fp32 weight + bias    *
 *      - int8 Conv1d:    int8 packed (DOTPROD) + fp32 scales + fp32 bias   *
 *      - int8 Linear:    int8 packed (DOTPROD) + fp32 scales [+ fp32 bias] *
 *      - GRU:            int8 W_ih + scales + bias + int8 W_hh + ...     *
 *      - PE (block 0):   fp32 [F2, C2]                                   *
 *  Int8 bytes are copied into an owning q_buf (with optional I8MM      *
 *  repack); fp32 weights/biases/PE are zero-copy pointers into `data`.  *
 *  The blob must outlive use.                                          */
static const char FE_INT8_MAGIC[8] = { 'F','M','_','W','8','_','0','3' };

static int fe_load_weights_int8(FeWeights *w, const void *data, size_t size) {
    const int D  = FE_GRU_DIM;
    const int C1 = FE_C1, C2 = FE_C2, F1 = FE_F1, F2 = FE_F2;
    const int enc_pre_K = 2 * FE_STRIDED_CI;        /* 16 = k2 * Ci after reshape */
    const int enc_pre_w_cnt = C1 * FE_STRIDED_CI * (FE_ENC_K0 / FE_STRIDE);
    const int dec_post_up_in = C1, dec_post_up_out = 2, dec_post_up_k = FE_ENC_K0;
    const int dec_post_up_w_cnt = dec_post_up_in * dec_post_up_out * dec_post_up_k;
    /* ---------- 1. Plan owning buffer sizes ---------- */
    size_t q_bytes  = 0;
    size_t s_floats = 0;

    /* Helper macro to accumulate for one int8 layer of shape [N, K]. */
    #define ACC_INT8(N_, K_) do { \
        q_bytes  += fe_qgemm_packed_size((K_), (N_)); \
        s_floats += (N_); \
    } while (0)

    /* enc[0..]: Conv1d k=3, in=C1, out=C1 -> [N=C1, K=C1*3] */
    for (int i = 0; i < FE_ENC_BLOCKS; ++i) ACC_INT8(C1, C1 * FE_ENC_K);
    /* rf_pre_lin */               ACC_INT8(F2, F1);
    /* rf_pre_conv */              ACC_INT8(C2, C1);
    for (int blk = 0; blk < FE_RF_BLOCKS; ++blk) {
        /* GRU W_ih, W_hh: prepacked int8 in the q8 blob. */
        ACC_INT8(3 * C2, C2);            /* GRU W_ih */
        ACC_INT8(3 * C2, C2);            /* GRU W_hh */
        ACC_INT8(C2, C2);                /* rnn_fc   */
        ACC_INT8(3 * C2, C2);            /* attn.qkv */
        ACC_INT8(C2, C2);                /* attn_fc  */
    }
    /* rf_post_lin */              ACC_INT8(F1, F2);
    /* rf_post_conv */             ACC_INT8(C1, C2);
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) {
        ACC_INT8(C1, 2 * C1);            /* dec_1x1 */
        ACC_INT8(C1, C1 * FE_ENC_K);     /* dec_3x3 */
    }
    /* dec_post_1x1 */             ACC_INT8(C1, 2 * C1);
    #undef ACC_INT8

    /* Fp32 prepack buffer:
     *   - GRU b_combined per block (3D each)
     *   - enc_pre.weight_packed_fp32 (fe_pack_W of [C1, 16])
     * GRU runtime is int8 W8A8 (see Wq_ih/Wq_hh below); the fp32-packed
     * sgemm form is dead and no longer allocated. */
    const size_t sz_enc_pre_pack =
        fe_sgemm_packed_size(enc_pre_K, C1) / sizeof(float);
    size_t prepack_floats =
        (size_t)FE_RF_BLOCKS * (3 * D) + sz_enc_pre_pack;

    /* Winograd F(2,3) storage: every k=3 conv layer (enc + dec_3x3) x 4
     * transformed weights each. Each is [N=C1, K=C1] (since enc/dec_3x3
     * are all C1 -> C1 with k=3) -- asserted below so a config change that
     * breaks either assumption fails at compile time rather than writing
     * past q_wino_buf in the derivation loop. */
    #define FE_K3_LAYERS (FE_ENC_BLOCKS + FE_DEC_BLOCKS)
    _Static_assert(FE_ENC_K == 3, "Winograd F(2,3) sizing assumes k=3");
    const size_t one_wino_bytes  = fe_qgemm_packed_size(C1, C1);
    const size_t wino_total_bytes = (size_t)FE_K3_LAYERS * 4 * one_wino_bytes;
    const size_t wino_total_scales = (size_t)FE_K3_LAYERS * 4 * C1;

    w->q_buf         = (signed char *)malloc(q_bytes);
    w->q_buf_size    = q_bytes;
    w->q_scales_buf  = (float *)malloc(s_floats * sizeof(float));
    w->q_scales_size = s_floats * sizeof(float);
    w->q_row_sums_buf  = (int32_t *)malloc(s_floats * sizeof(int32_t));
    w->q_row_sums_size = s_floats * sizeof(int32_t);
    w->q_wino_buf         = (signed char *)malloc(wino_total_bytes);
    w->q_wino_buf_size    = wino_total_bytes;
    w->q_wino_scales_buf  = (float *)malloc(wino_total_scales * sizeof(float));
    w->q_wino_scales_size = wino_total_scales * sizeof(float);
    w->prepack_buf   = (float *)malloc(prepack_floats * sizeof(float));
    w->prepack_size  = prepack_floats * sizeof(float);
    if (!w->q_buf || !w->q_scales_buf || !w->q_row_sums_buf
        || !w->q_wino_buf || !w->q_wino_scales_buf || !w->prepack_buf)
        return -3;

    /* ---------- 2. Walk file payload, set up pointers ---------- */
    const char  *blob   = (const char *)data;
    size_t       off    = 12;                  /* skip 8B magic + u32 version */

    signed char *q_cur  = w->q_buf;
    float       *s_cur  = w->q_scales_buf;
    int32_t     *r_cur  = w->q_row_sums_buf;
    float       *pre_cur = w->prepack_buf;

    /* take_fp32(): zero-copy fp32 pointer into blob. */
    #define TAKE_FP32(count_) ({                          \
        const float *_p = (const float *)(blob + off);    \
        off += (size_t)(count_) * sizeof(float);           \
        if (off > size) return -1;                         \
        _p;                                                \
    })

    /* take_int8_layer(): copy int8 packed weight + scales [+ optional bias]
     * for one [N, K] layer. Sets *q_out, *s_out, *r_out, *b_out (b_out can
     * be NULL). row_sums are computed from the packed int8 buffer (DOTPROD
     * layout) before any potential I8MM repack. */
    #define TAKE_INT8_LAYER(N_, K_, q_out_, s_out_, r_out_, b_out_) do { \
        int _Nv = (N_), _Kv = (K_);                        \
        size_t _qb = fe_qgemm_packed_size(_Kv, _Nv);       \
        if (off + _qb > size) return -1;                    \
        memcpy(q_cur, blob + off, _qb); off += _qb;         \
        if (off + (size_t)_Nv * sizeof(float) > size) return -1; \
        memcpy(s_cur, blob + off, (size_t)_Nv * sizeof(float)); \
        off += (size_t)_Nv * sizeof(float);                 \
        fe_qgemm_compute_row_sums(q_cur, _Nv, _Kv, r_cur);  \
        *(q_out_) = q_cur;                                  \
        *(s_out_) = s_cur;                                  \
        *(r_out_) = r_cur;                                  \
        if (b_out_) {                                       \
            if (off + (size_t)_Nv * sizeof(float) > size) return -1; \
            *(b_out_) = (const float *)(blob + off);        \
            off += (size_t)_Nv * sizeof(float);             \
        }                                                   \
        /* Layout finalize for I8MM tier happens via fe_weights_finalize_for_tier()
         * after fe_qgemm_init() resolves the runtime tier. */          \
        q_cur += _qb;                                       \
        s_cur += _Nv;                                       \
        r_cur += _Nv;                                       \
    } while (0)

    /* ---- enc_pre (fp32-only inside the q8 blob) ---- */
    w->enc_pre.out_ch = C1;
    w->enc_pre.kernel = FE_ENC_K0 / FE_STRIDE;
    w->enc_pre.in_ch = FE_STRIDE * 2;
    const float *enc_pre_weight = TAKE_FP32(enc_pre_w_cnt);
    w->enc_pre.bias   = TAKE_FP32(C1);
    w->enc_pre.weight_q   = NULL;
    w->enc_pre.scales_w   = NULL;
    w->enc_pre.row_sums   = NULL;
    /* Pack enc_pre fp32 weight for fe_strided_conv1d's weight_packed_fp32. */
    fe_pack_W(enc_pre_weight, C1, enc_pre_K, pre_cur);
    w->enc_pre.weight_packed_fp32 = pre_cur;
    pre_cur += sz_enc_pre_pack;
    fe_actscale_init(&w->enc_pre.act);

    /* ---- enc[0..ENC_BLOCKS-1] (int8) ---- */
    for (int i = 0; i < FE_ENC_BLOCKS; ++i) {
        FeConv1d *c = &w->enc[i];
        c->in_ch = C1; c->out_ch = C1; c->kernel = FE_ENC_K;
        c->weight_packed_fp32 = NULL;
        TAKE_INT8_LAYER(C1, C1 * FE_ENC_K, &c->weight_q, &c->scales_w, &c->row_sums, &c->bias);
        fe_actscale_init(&c->act);
    }

    /* ---- rf_pre_lin (int8, no bias) ---- */
    w->rf_pre_lin.in_dim = F1; w->rf_pre_lin.out_dim = F2;
    {
        const float *_no_bias = NULL;
        TAKE_INT8_LAYER(F2, F1, &w->rf_pre_lin.weight_q, &w->rf_pre_lin.scales_w, &w->rf_pre_lin.row_sums, (const float **)NULL);
        w->rf_pre_lin.bias = _no_bias;
    }
    fe_actscale_init(&w->rf_pre_lin.act); fe_actscale_init(&w->rf_pre_lin.act_out);

    /* ---- rf_pre_conv (int8, k=1) ---- */
    w->rf_pre_conv.in_ch = C1; w->rf_pre_conv.out_ch = C2; w->rf_pre_conv.kernel = 1;
    w->rf_pre_conv.weight_packed_fp32 = NULL;
    TAKE_INT8_LAYER(C2, C1, &w->rf_pre_conv.weight_q, &w->rf_pre_conv.scales_w, &w->rf_pre_conv.row_sums, &w->rf_pre_conv.bias);
    fe_actscale_init(&w->rf_pre_conv.act);

    /* ---- RNNFormer blocks ---- */
    for (int blk = 0; blk < FE_RF_BLOCKS; ++blk) {
        FeRNNFormerBlock *rb = &w->rf[blk];
        FeGRU *g = &rb->gru;
        g->hidden_size = D;
        TAKE_INT8_LAYER(3 * D, D, &g->Wq_ih, &g->scales_ih,
                        &g->row_sums_ih, &g->b_ih);
        TAKE_INT8_LAYER(3 * D, D, &g->Wq_hh, &g->scales_hh,
                        &g->row_sums_hh, &g->b_hh);
        /* b_combined = b_ih + b_hh */
        for (int j = 0; j < 3 * D; ++j) pre_cur[j] = g->b_ih[j] + g->b_hh[j];
        g->b_combined = pre_cur; pre_cur += 3 * D;
        fe_actscale_init(&g->act_x); fe_actscale_init(&g->act_h);

        /* rnn_fc */
        rb->rnn_fc.in_dim = C2; rb->rnn_fc.out_dim = C2;
        TAKE_INT8_LAYER(C2, C2, &rb->rnn_fc.weight_q, &rb->rnn_fc.scales_w, &rb->rnn_fc.row_sums, &rb->rnn_fc.bias);
        fe_actscale_init(&rb->rnn_fc.act); fe_actscale_init(&rb->rnn_fc.act_out);

        /* attn.qkv (no bias) */
        rb->attn.qkv.in_dim = C2; rb->attn.qkv.out_dim = 3 * C2;
        rb->attn.qkv.bias = NULL;
        TAKE_INT8_LAYER(3 * C2, C2, &rb->attn.qkv.weight_q, &rb->attn.qkv.scales_w, &rb->attn.qkv.row_sums, (const float **)NULL);
        fe_actscale_init(&rb->attn.qkv.act); fe_actscale_init(&rb->attn.qkv.act_out);

        /* attn_fc */
        rb->attn_fc.in_dim = C2; rb->attn_fc.out_dim = C2;
        TAKE_INT8_LAYER(C2, C2, &rb->attn_fc.weight_q, &rb->attn_fc.scales_w, &rb->attn_fc.row_sums, &rb->attn_fc.bias);
        fe_actscale_init(&rb->attn_fc.act); fe_actscale_init(&rb->attn_fc.act_out);

        if (blk == 0) {
            rb->pe     = TAKE_FP32(F2 * C2);
            rb->has_pe = 1;
        } else {
            rb->pe     = NULL;
            rb->has_pe = 0;
        }
    }

    /* ---- rf_post_lin (int8, no bias) ---- */
    w->rf_post_lin.in_dim = F2; w->rf_post_lin.out_dim = F1;
    w->rf_post_lin.bias = NULL;
    TAKE_INT8_LAYER(F1, F2, &w->rf_post_lin.weight_q, &w->rf_post_lin.scales_w, &w->rf_post_lin.row_sums, (const float **)NULL);
    fe_actscale_init(&w->rf_post_lin.act); fe_actscale_init(&w->rf_post_lin.act_out);

    /* ---- rf_post_conv (int8) ---- */
    w->rf_post_conv.in_ch = C2; w->rf_post_conv.out_ch = C1; w->rf_post_conv.kernel = 1;
    w->rf_post_conv.weight_packed_fp32 = NULL;
    TAKE_INT8_LAYER(C1, C2, &w->rf_post_conv.weight_q, &w->rf_post_conv.scales_w, &w->rf_post_conv.row_sums, &w->rf_post_conv.bias);
    fe_actscale_init(&w->rf_post_conv.act);

    /* ---- Decoder blocks ---- */
    for (int i = 0; i < FE_DEC_BLOCKS; ++i) {
        FeConv1d *c1 = &w->dec_1x1[i];
        c1->in_ch = 2 * C1; c1->out_ch = C1; c1->kernel = 1;
        c1->weight_packed_fp32 = NULL;
        TAKE_INT8_LAYER(C1, 2 * C1, &c1->weight_q, &c1->scales_w, &c1->row_sums, &c1->bias);
        fe_actscale_init(&c1->act);

        FeConv1d *c3 = &w->dec_3x3[i];
        c3->in_ch = C1; c3->out_ch = C1; c3->kernel = FE_ENC_K;
        c3->weight_packed_fp32 = NULL;
        TAKE_INT8_LAYER(C1, C1 * FE_ENC_K, &c3->weight_q, &c3->scales_w, &c3->row_sums, &c3->bias);
        fe_actscale_init(&c3->act);
    }

    /* ---- dec_post_1x1 (int8) ---- */
    w->dec_post_1x1.in_ch = 2 * C1; w->dec_post_1x1.out_ch = C1; w->dec_post_1x1.kernel = 1;
    w->dec_post_1x1.weight_packed_fp32 = NULL;
    TAKE_INT8_LAYER(C1, 2 * C1, &w->dec_post_1x1.weight_q, &w->dec_post_1x1.scales_w, &w->dec_post_1x1.row_sums, &w->dec_post_1x1.bias);
    fe_actscale_init(&w->dec_post_1x1.act);

    /* ---- dec_post_up (ConvT1d, fp32-only) ---- */
    w->dec_post_up.in_ch = dec_post_up_in;
    w->dec_post_up.out_ch = dec_post_up_out;
    w->dec_post_up.kernel = dec_post_up_k;
    w->dec_post_up.stride = FE_STRIDE;
    w->dec_post_up.weight = TAKE_FP32(dec_post_up_w_cnt);
    w->dec_post_up.bias   = TAKE_FP32(dec_post_up_out);

    if (off != size) return -1;

    /* ---------- 3. Derive Winograd F(2,3) transformed weights ---------- *
     * For each k=3 conv layer, dequantise the packed int8 weight back to
     * fp32, apply G*W*G^T to produce 4 transformed [Co, Ci] matrices,
     * then symmetric-int8 quantise + pack each. The packed Wg buffers are
     * laid out contiguously in q_wino_buf, scales in q_wino_scales_buf.   */
    {
        signed char *wq_cur = w->q_wino_buf;
        float       *ws_cur = w->q_wino_scales_buf;
        /* Worst-case scratch: Co*K + 4*Co*Ci floats = C1*(3+4)*C1 = 7*C1².
         * Init-only memory (Winograd weight derivation runs once). Was a
         * function-local static (252 KiB permanent BSS); now malloc'd and
         * freed after the derivation loop, returning the memory to the
         * heap. */
        const size_t wino_scratch_floats = (size_t)7 * FE_C1 * FE_C1;
        float *wino_scratch = (float *)malloc(wino_scratch_floats * sizeof(float));
        if (!wino_scratch) return -1;

        FeConv1d *k3_layers[FE_ENC_BLOCKS + FE_DEC_BLOCKS];
        int n_k3 = collect_k3_layers(w, k3_layers);

        for (int i = 0; i < n_k3; ++i) {
            FeConv1d *cv = k3_layers[i];
            int Co = cv->out_ch;
            int Ci = cv->in_ch;
            signed char *packed_out[4];
            float       *scales_out[4];
            for (int j = 0; j < 4; ++j) {
                packed_out[j] = wq_cur;  wq_cur += fe_qgemm_packed_size(Ci, Co);
                scales_out[j] = ws_cur;  ws_cur += Co;
            }
            fe_winograd_f23_derive_weights(cv->weight_q, cv->scales_w,
                                            Co, Ci,
                                            packed_out, scales_out,
                                            wino_scratch);
            for (int j = 0; j < 4; ++j) {
                cv->wino_weight_q[j] = packed_out[j];
                cv->wino_scales[j]   = scales_out[j];
            }
        }
        free(wino_scratch);
    }

    #undef TAKE_FP32
    #undef TAKE_INT8_LAYER
    return 0;
}

/* ------------------------------------------------------------------ */
int fe_load_weights(FeWeights *w, const void *data, size_t size) {
    if (size >= 12 && memcmp(data, FE_INT8_MAGIC, 8) == 0) {
        return fe_load_weights_int8(w, data, size);
    }
    (void)w;
    return -1;
}

// Convert all packed weights to the layout required by `tier`. The packer
// produces DOTPROD layout; I8MM tier requires an extra in-place repack per
// N-block. NEON / DOTPROD / x86 tiers accept DOTPROD layout unchanged.
// Must be called after fe_load_weights and after fe_qgemm_init has resolved
// the runtime tier.
void fe_weights_finalize_for_tier(FeWeights *w, int tier) {
    if (!w) return;
    if (tier != FE_QGEMM_TIER_ARM_I8MM) return;  // DOTPROD layout is the default.

    const int D = FE_GRU_DIM;

    FeLinear *linears[FE_RF_BLOCKS * 3 + 2];
    int n_lin = collect_linears(w, linears);

    for (int i = 0; i < n_lin; ++i) {
        FeLinear *lin = linears[i];
        fe_qgemm_repack_i8mm((int8_t *)lin->weight_q, lin->out_dim, lin->in_dim);
    }
    /* GRU is int8 (W8A8 with asymmetric uint8 activations) -- repack both
     * Wq_ih and Wq_hh for I8MM tier. K=D=hidden_size is a multiple of 8
     * for our configs so the repack runs cleanly. */
    for (int blk = 0; blk < FE_RF_BLOCKS; ++blk) {
        FeGRU *g = &w->rf[blk].gru;
        if (g->Wq_ih)
            fe_qgemm_repack_i8mm((int8_t *)g->Wq_ih, 3 * D, D);
        if (g->Wq_hh)
            fe_qgemm_repack_i8mm((int8_t *)g->Wq_hh, 3 * D, D);
    }

    /* Winograd F(2,3) per-layer transformed weights also live in DOTPROD
     * layout (fe_qgemm_pack_W output); convert to I8MM layout for the
     * inner GEMMs. K=Ci on these -- for k=3 enc/dec all are C1=96 = 12*8,
     * a clean multiple of 8 so the repack runs without partial blocks. */
    FeConv1d *k3_layers[FE_ENC_BLOCKS + FE_DEC_BLOCKS];
    int n_k3 = collect_k3_layers(w, k3_layers);
    for (int i = 0; i < n_k3; ++i) {
        FeConv1d *cv = k3_layers[i];
        int Co = cv->out_ch;
        int Ci = cv->in_ch;
        for (int j = 0; j < 4; ++j) {
            if (cv->wino_weight_q[j])
                fe_qgemm_repack_i8mm((int8_t *)cv->wino_weight_q[j], Co, Ci);
        }
    }

    FeConv1d *convs[2 + FE_ENC_BLOCKS + FE_DEC_BLOCKS * 2 + 1 + 1];
    int n_conv = collect_convs(w, convs);
    for (int i = 0; i < n_conv; ++i) {
        FeConv1d *cv = convs[i];
        if (!cv->weight_q) continue;  // enc_pre uses fp32-only path.
        int Co = cv->out_ch;
        int Ki = cv->in_ch * cv->kernel;
        fe_qgemm_repack_i8mm((int8_t *)cv->weight_q, Co, Ki);
    }
}

void fe_free_weights(FeWeights *w) {
    if (!w) return;
    if (w->q_buf) { free(w->q_buf); w->q_buf = NULL; w->q_buf_size = 0; }
    if (w->q_scales_buf) { free(w->q_scales_buf); w->q_scales_buf = NULL; w->q_scales_size = 0; }
    if (w->q_row_sums_buf) { free(w->q_row_sums_buf); w->q_row_sums_buf = NULL; w->q_row_sums_size = 0; }
    if (w->q_wino_buf) { free(w->q_wino_buf); w->q_wino_buf = NULL; w->q_wino_buf_size = 0; }
    if (w->q_wino_scales_buf) { free(w->q_wino_scales_buf); w->q_wino_scales_buf = NULL; w->q_wino_scales_size = 0; }
    if (w->prepack_buf) {
        free(w->prepack_buf);
        w->prepack_buf  = NULL;
        w->prepack_size = 0;
    }
}

// Weight descriptors. BatchNorm pre-fused into conv/linear weights at ONNX
// export time. Runtime accepts only the FM_W8_03 q8 blob layout fixed by
// tools/quantize_bin.py.
#ifndef FE_WEIGHTS_H
#define FE_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>
#include "fe_config_medium.h"
#include "fe_calib.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Conv1d (BatchNorm pre-fused -> bias present). Weight: [out, in, k]. */
typedef struct {
    const float *bias;
    int in_ch, out_ch, kernel;
    /* Q8 path: weight packed as Linear-style [N=Co, K=Ci*k] int8 +
     * per-channel scales. Used directly by Conv k=1; k=3 also keeps this
     * source layout so load-time finalization can derive Winograd weights. */
    const signed char *weight_q;
    const float       *scales_w;
    /* Σ_k weight_q[n, k] per output row -- used for the (128 - zp_x)
     * correction term in the asymmetric uint8 activation dequant. */
    const int32_t     *row_sums;
    /* fp32 NR-blocked pack (for paths that use fe_sgemm, e.g. enc_pre
     * strided conv). NULL for layers that don't need it.                */
    const float       *weight_packed_fp32;
    /* Winograd F(2,3) transformed weights for k=3 1-D conv.
     * G*W*G^T produces 4 [Co, Ci] matrices, each independently quantised
     * with its own per-row int8 scale. Only populated for kernel == 3
     * convolutions; NULL for k=1 / strided. The inner I8MM GEMM consumes
     * these directly (symmetric int8, no row-sum correction needed since
     * the transformed-input activation quant uses zp=128).                */
    const signed char *wino_weight_q[4];
    const float       *wino_scales[4];
    /* Activation running min/max diagnostics; quant scale is per-frame. */
    FeActScale         act;
} FeConv1d;

/* ConvTranspose1d. Weight layout (PyTorch convention): [in, out, k]. */
typedef struct {
    const float *weight;
    const float *bias;
    int in_ch, out_ch, kernel, stride;
} FeConvT1d;

/* Linear y = W @ x + b. bias may be NULL.
 * weight_q:      int8 quantized weight in DOTPROD-friendly layout
 *                (size = align_up(out_dim, NR) x in_dim bytes)
 * scales_w:      [out_dim] fp32 per-channel scales for dequant       */
typedef struct {
    const float *bias;
    const signed char *weight_q;
    const float *scales_w;
    const int32_t *row_sums;        /* Σ_k weight_q[n, k] per output row    */
    int in_dim, out_dim;
    /* Input activation running min/max diagnostics. */
    FeActScale  act;
    FeActScale  act_out;     /* output post-bias post-silu -- for int8 chain */
} FeLinear;

/* GRU (PyTorch convention, linear_before_reset=1, gates [r, z, n]).
 *
 * Pre-pack:
 *   b_combined [3D] = b_ih + b_hh   -- fused once per load; r/z gates use
 *                                     it directly (saves one add per
 *                                     element per gate per frame).
 *   (A single fused W is mathematically *more* FLOPs than 2 GEMMs for
 *    this model because the n gate needs hh and ih separately --
 *    fusing introduces a redundant n recomputation -- so we keep the
 *    two original GEMMs.)                                              */
typedef struct {
    const float *b_ih;       /* [3*hidden]                                 */
    const float *b_hh;       /* [3*hidden]                                 */
    const float *b_combined; /* [3*hidden] owning, = b_ih + b_hh          */
    const signed char *Wq_ih;   /* int8 quantized W_ih, DOTPROD layout       */
    const signed char *Wq_hh;   /* int8 quantized W_hh                    */
    const float       *scales_ih;  /* [3*hidden] per-row scale of W_ih    */
    const float       *scales_hh;  /* [3*hidden] per-row scale of W_hh    */
    const int32_t     *row_sums_ih;  /* [3*hidden] Σ_k W_ih_i8[n,k]       */
    const int32_t     *row_sums_hh;  /* [3*hidden] Σ_k W_hh_i8[n,k]       */
    int hidden_size;
    /* GRU has TWO distinct input activations (x and h-prev), so it carries
     * two independent running range trackers. Quant scales are per-frame. */
    FeActScale  act_x;
    FeActScale  act_h;
} FeGRU;

/* Multi-head self-attention with fused QKV. */
typedef struct {
    FeLinear qkv;          /* [3*C, C] (no bias for this model) */
} FeAttention;

/* Single RNNFormer block */
typedef struct {
    FeGRU       gru;
    FeLinear    rnn_fc;        /* [C2, C2] bias=True (fused BN) */
    FeAttention attn;
    FeLinear    attn_fc;       /* [C2, C2] bias=True (fused BN) */
    const float *pe;           /* [F2, C2] positional embedding (block 0 only) */
    int         has_pe;
} FeRNNFormerBlock;

/* Full model weights */
typedef struct {
    /* Encoder PreNet: StridedConv(2->C1, k=8, s=4)
     * -- stored as Conv1d(8->C1, k=2) after reshape trick */
    FeConv1d  enc_pre;

    /* Encoder blocks x ENC_BLOCKS (k=3) */
    FeConv1d  enc[FE_ENC_BLOCKS];

    /* RNNFormer PreNet */
    FeLinear  rf_pre_lin;      /* [F2, F1] no bias */
    FeConv1d  rf_pre_conv;     /* C1 -> C2, k=1     */

    /* RNNFormer blocks */
    FeRNNFormerBlock rf[FE_RF_BLOCKS];

    /* RNNFormer PostNet */
    FeLinear  rf_post_lin;     /* [F1, F2] no bias */
    FeConv1d  rf_post_conv;    /* C2 -> C1, k=1     */

    /* Decoder */
    FeConv1d  dec_1x1[FE_DEC_BLOCKS]; /* concat 2*C1 -> C1, k=1 */
    FeConv1d  dec_3x3[FE_DEC_BLOCKS]; /* C1 -> C1, k=3          */

    /* Decoder PostNet */
    FeConv1d  dec_post_1x1;    /* concat 2*C1 -> C1, k=1 */
    FeConvT1d dec_post_up;     /* C1 -> 2, k=8, s=4      */

    /* Owning storage for init-derived fp32 helpers (GRU b_combined and
     * enc_pre.weight_packed_fp32).
     * *_size fields below are byte counts, matching malloc/free/try_mlock. */
    float *prepack_buf;
    size_t prepack_size;

    /* Owning storage for int8 quantized Linear weights + per-channel scales. */
    signed char *q_buf;
    size_t       q_buf_size;
    float       *q_scales_buf;
    size_t       q_scales_size;
    /* Owning storage for per-row int8 weight sums (Σ_k W_q[n,k], int32).
     * Sized to match q_scales_buf (one entry per output channel across all
     * int8 layers). Populated at load time by fe_qgemm_compute_row_sums. */
    int32_t     *q_row_sums_buf;
    size_t       q_row_sums_size;
    /* Winograd F(2,3) transformed weights: 4 int8 [Co, Ci] packed matrices
     * per k=3 conv layer + their per-row fp32 scales. Allocated only when
     * there is at least one k=3 conv; encoder + decoder k3 share this buf. */
    signed char *q_wino_buf;
    size_t       q_wino_buf_size;
    float       *q_wino_scales_buf;
    size_t       q_wino_scales_size;
} FeWeights;

/* Parse an FM_W8_03 q8 blob into weight pointers. The blob must outlive use. */
int fe_load_weights(FeWeights *w, const void *data, size_t size);

/* Convert packed-weight layouts to whatever the chosen runtime GEMM tier
 * expects (currently: I8MM needs an extra repack, every other tier accepts
 * the default DOTPROD layout). Call after fe_load_weights + fe_qgemm_init. */
void fe_weights_finalize_for_tier(FeWeights *w, int tier);

/* Free prepacked storage (does NOT touch the input blob).                 */
void fe_free_weights(FeWeights *w);

#ifdef __cplusplus
}
#endif

#endif /* FE_WEIGHTS_H */

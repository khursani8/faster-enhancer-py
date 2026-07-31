/*
 * fe_config_medium.h -- FastEnhancer-Medium 48 kHz compile-time constants.
 *
 * External source config: configs/fastenhancer_48khz/m.yaml
 *   (from the FastEnhancer reference tree; not vendored in this repo)
 *   channels (C1)            = 96
 *   ENC_BLOCKS               = 3    (kernel_size [8, 3, 3, 3])
 *   rnnformer.channels (C2)  = 72
 *   rnnformer.freq    (F2)   = 72
 *   RF_BLOCKS                = 4
 *   num_heads                = 4    (head_dim = 18)
 *   n_fft                    = 1024
 *   hop_size                 = 320  (6.67 ms @ 48 kHz)
 *   win_size                 = 1024
 *   input_compression        = 0.3
 *
 * Total params: 511,754; fp32-equivalent bytes: 2,047,016.
 * Direct-conv-equivalent multiply footprint: 53.4 MMAC/frame.
 * Dense-equivalent runtime multiply footprint with Winograd k=3 convs:
 * 46.3 MMAC/frame, including the fp32 enc_pre / dec_post_up edge convs
 * (~6.95 GMAC/s at 150 frames/s).
 */
#ifndef FE_CONFIG_MEDIUM_H
#define FE_CONFIG_MEDIUM_H

#include "fe.h"   /* FE_SAMPLE_RATE, FE_FRAME_SIZE (public ABI constants) */

/* Audio */
#define FE_N_FFT         1024
#define FE_HOP_SIZE      FE_FRAME_SIZE  /* 320, 6.67 ms */
#define FE_WIN_SIZE      1024
#define FE_FREQ_BINS     (FE_N_FFT / 2)         /* 512 */
#define FE_SPEC_BINS     (FE_N_FFT / 2 + 1)     /* 513 */
#define FE_CACHE_LEN     (FE_N_FFT - FE_HOP_SIZE) /* 704 */

/* Compression (input/output dynamic range) */
#define FE_COMPRESS_EXP  0.3f
#define FE_COMPRESS_IN   (FE_COMPRESS_EXP - 1.0f)        /* -0.7  */
#define FE_COMPRESS_OUT  (1.0f / FE_COMPRESS_EXP - 1.0f) /*  2.333 */

/* Encoder / Decoder */
#define FE_STRIDE        4
#define FE_ENC_K0        8     /* enc_pre kernel */
#define FE_ENC_K         3     /* encoder block kernel */
#define FE_C1            96    /* encoder/decoder channels */
#define FE_ENC_BLOCKS    3
#define FE_DEC_BLOCKS    FE_ENC_BLOCKS

/* RNNFormer */
#define FE_C2            72    /* rf channels */
#define FE_F2            72    /* rf freq */
#define FE_RF_BLOCKS     4
#define FE_NUM_HEADS     4
#define FE_HEAD_DIM      (FE_C2 / FE_NUM_HEADS)  /* 18 */

/* HEAD_DIM padding for the int8 attention path.
 *   HD_PAD_K = next multiple of 4   (DOTPROD K-axis alignment)
 *   HD_PAD_V = next multiple of NR=8 (V S@V N-axis alignment)
 * For Medium HD=18 -> HD_PAD_K=20, HD_PAD_V=24. Padding lanes are zero
 * so they contribute nothing to dot products.
 *
 * NB: padding HD_PAD_K up to 8 (so I8MM fires instead of DOTPROD for Q@K^T)
 * was prototyped and measured slower on M2 -- DOTPROD issues at 4/cycle while
 * I8MM throughput is ~2/cycle, and for the tiny Q@K^T (M=N=72, K=20) the
 * DOTPROD instruction parallelism beats I8MM's higher MAC density. The K-pack
 * repack added ~1.7 µs/frame of pure overhead on top. Left at 4 alignment. */
#define FE_HEAD_DIM_PAD_K  (((FE_HEAD_DIM + 3) / 4) * 4)
#define FE_HEAD_DIM_PAD_V  (((FE_HEAD_DIM + 7) / 8) * 8)

#define FE_GRU_DIM       FE_C2
#define FE_GRU_GATES     3                       /* r, z, n */

/* Derived */
#define FE_F1            (FE_FREQ_BINS / FE_STRIDE) /* 128 */

/* StridedConv reshape (input [2, F=512] -> [8, F1+1=129], Conv1d(8->C1, k=2)) */
#define FE_STRIDED_CI    (FE_STRIDE * 2)         /* 8   */
#define FE_STRIDED_FNEW  (FE_F1 + 1)             /* 129 */

/* FE_FRAME_SIZE is defined publicly in fe.h (= 320). FE_HOP_SIZE above
 * aliases it; both compile-time equal. */

/* ---- GEMM tile-alignment contract ----
 * The qgemm kernels were stripped of their scalar M%MR / N%NR remainder
 * path (see fe_qgemm_tail_unsupported in arch_kernels.h / qgemm_dispatch.c).
 * That is only safe while every GEMM M and N dimension is an exact multiple
 * of the 8x8 microtile (FE_QGEMM_MR == FE_QGEMM_NR == 8). Those dimensions are
 * the channel / freq / padded-head constants below, so assert them here: a
 * future edit to a non-multiple-of-8 value fails the build loudly instead of
 * silently hitting the removed remainder path. (HD_PAD_V is the attention V
 * N-axis; its round-up-to-8 padding formula is what keeps S@V tile-aligned.) */
_Static_assert(FE_C1            % 8 == 0, "FE_C1 must be a multiple of the 8x8 GEMM tile");
_Static_assert(FE_C2            % 8 == 0, "FE_C2 must be a multiple of the 8x8 GEMM tile");
_Static_assert(FE_F2            % 8 == 0, "FE_F2 must be a multiple of the 8x8 GEMM tile");
_Static_assert(FE_F1            % 8 == 0, "FE_F1 must be a multiple of the 8x8 GEMM tile");
_Static_assert(FE_HEAD_DIM_PAD_V % 8 == 0, "FE_HEAD_DIM_PAD_V must be a multiple of the 8x8 GEMM tile");

#endif /* FE_CONFIG_MEDIUM_H */

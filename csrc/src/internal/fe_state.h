/*
 * Runtime buffers for the streaming engine. One calloc at fe_init, no
 * heap traffic during inference. Default layout is [F, C] (frequency
 * first) so qgemm input rows fall out of conv1d k=1, k=3, residual adds
 * and skip-concats without a transpose.
 *
 * Exceptions: frequency-axis linears fold [F, C] -> [C, F] access into
 * the quantize pass, and MHSA strides over the [F2, 3*C2] qkv buffer
 * directly.
 */
#ifndef FE_STATE_H
#define FE_STATE_H

#include <stdint.h>
#include <assert.h>
#include "fe_config_medium.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Encoder ping-pong: Medium has FE_ENC_BLOCKS=3 (odd), so the last
 * encoder k=3 write lands in buf_b. The RF pre-net consumes the fp16
 * enc_skip path, and rf_post_conv later seeds decoder input in buf_a.
 */
typedef struct {
	/* STFT / iSTFT overlap caches (704 samples each). */
	float cache_stft[FE_CACHE_LEN];
	float cache_istft[FE_CACHE_LEN];

	float window[FE_N_FFT];
	float window_istft[FE_N_FFT];   /* window / sum_of_squares / N */

	/*
	 * fft_buf is live during STFT/iSTFT; mask_cf is live during the
	 * mask multiply. Disjoint -> union.
	 */
	union {
		float fft_buf[FE_N_FFT];            /* STFT/iSTFT scratch */
		float mask_cf[2 * FE_FREQ_BINS];    /* split-layout mask  */
	};

	/* Output spectrum in split layout [re | im]. */
	float spec_out[FE_FREQ_BINS * 2];

	/*
	 * Two aliasing unions for the enc/dec/RF working buffers:
	 *   buf_a / rf_b      - enc/dec working vs. RF block working
	 *   buf_b / rf_lin_T  - enc/dec ping-pong vs. rf_pre/post output
	 * The RF stage and the enc/dec stage never overlap.
	 */
	union {
		float buf_a[FE_C1 * FE_F1];
		float rf_b [FE_F2 * FE_C2];
	};
	union {
		float buf_b   [FE_C1 * FE_F1];
		float rf_lin_T[FE_C2 * FE_F1];
	};

	/*
	 * Encoder skip connections — IEEE fp16 storage. int8 storage
	 * was tried and lost ~40 dB SNR through dec_1x1's joint min/max
	 * round-trip; fp16 keeps mantissa 10 bits (~3 decimal digits) and
	 * is validated by the quality suite.
	 *
	 * Kernel-side fp16 read/write — Winograd epilogue and
	 * fe_silu_skip_fp16 store fp16 directly via VCVTPS2PH / FCVTN;
	 * concat2_fp16b / transposed_in_fp16 read fp16 with on-the-fly cvt.
	 * enc_skip_scratch (49,152 B / 48 KiB fp32 staging area) eliminated.
	 * enc_skip fp16 storage saves 98,304 B (96 KiB) vs fp32 storage.
	 */
	uint16_t enc_skip[FE_ENC_BLOCKS + 1][FE_C1 * FE_F1];

	/*
	 * MHSA and Winograd scratch share storage: MHSA is hot only inside
	 * RF blocks, Winograd only inside enc/dec k=3 convs.
	 */
	union {
		struct {
			float attn_qkv   [FE_F2 * 3 * FE_C2];
			float attn_scores[FE_F2];        /* one softmax row */
			float attn_out   [FE_F2 * FE_C2];

			int8_t qkv_q       [FE_F2 * 3 * FE_C2];
			int8_t attn_Qq     [FE_NUM_HEADS * FE_F2 * FE_HEAD_DIM_PAD_K];
			int8_t attn_Kpq    [FE_NUM_HEADS * FE_F2 * FE_HEAD_DIM_PAD_K];
			int8_t attn_Vpq    [FE_NUM_HEADS * FE_F2 * FE_HEAD_DIM_PAD_V];
			int8_t attn_scoresq[FE_F2 * FE_F2];
		};
		struct {
			/* F(2,3) Winograd: 4 int32 inner-GEMM tiles + 4 int8
			 * transformed-input planes. */
			int32_t wino_U [4 * (FE_F1 / 2) * FE_C1];
			int8_t  wino_dq[4 * (FE_F1 / 2) * FE_C1];
		};
	};

	/*
	 * GRU hidden state per RF block, persistent across frames.
	 * fp16 storage. The fused GRU kernel reads h_old directly from fp16,
	 * then dual-stores h_new to this fp16 buffer and to a stack fp32
	 * scratch consumed by rnn_fc.
	 */
	uint16_t gru_h[FE_RF_BLOCKS][FE_F2 * FE_C2];

	/*
	 * Shared qgemm scratch. Sized to the worst-case across live paths:
	 *   aq  - dec_1x1 concat quantize, F1 * 2*C1 bytes
	 *   c32 - MHSA Q@K^T, F2 * F2 ints
	 */
	int8_t  qgemm_aq [FE_F1 * 2 * FE_C1];
	int32_t qgemm_c32[FE_F2 * FE_F2];

	/* STFT deinterleave: row 0 = real lanes, row 1 = imag lanes. */
	float spec_cf[2 * FE_FREQ_BINS];
} FeState;

#ifdef __cplusplus
}
#endif

#endif /* FE_STATE_H */

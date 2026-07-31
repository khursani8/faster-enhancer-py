/*
 * fe.h — Public C API for faster-enhancer.c, a 48 kHz speech-denoiser engine
 *        (FastEnhancer-Medium, W8A8 int8, C ABI + SIMD intrinsics).
 *
 *   One inference call processes FE_FRAME_SIZE input samples at 48 kHz
 *   (320 samples = 6.67 ms) and produces FE_FRAME_SIZE output samples.
 *   The streaming STFT is zero-look-ahead but has a fixed 704-sample
 *   (14.67 ms) alignment delay.
 *
 *   Streaming usage:
 *       fe_init(weights_blob, weights_size);
 *       for each 320-sample frame:
 *           fe_run(in, out);
 *       fe_free();
 *
 *   Runtime contract:
 *       single global engine instance; single-threaded calls only.
 *       Do not call fe_init/run/reset/free concurrently. Run fe_run()
 *       from one audio/render thread, and use external synchronization if an
 *       application needs to hand ownership between threads.
 */
#ifndef FE_H
#define FE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- *
 *  Compile-time constants (part of the public ABI contract).
 * ---------------------------------------------------------------- */
#ifndef FE_SAMPLE_RATE
#  define FE_SAMPLE_RATE  48000     /* fixed: model is trained at 48 kHz */
#endif
#ifndef FE_FRAME_SIZE
#  define FE_FRAME_SIZE   320       /* samples per fe_run() call (6.67 ms) */
#endif

/* ---------------------------------------------------------------- *
 *  Lifecycle.
 * ---------------------------------------------------------------- */

/* Initialize the single global engine from a fe.q8 / FM_W8_03 q8 weights blob.
 * The blob must remain alive until fe_free(): fp32-only weights, biases,
 * and positional embedding are referenced zero-copy. Returns 0 on success,
 * non-zero on failure. A second call while initialized is a no-op; call
 * fe_free() before loading a different blob. Not thread-safe. */
int  fe_init (const void *weights_blob, int weights_size);

/* Release all engine state and free internal buffers. Safe to call more than
 * once; after this, fe_run() and fe_reset() are no-ops until fe_init()
 * succeeds again. Not thread-safe with fe_run(). */
void fe_free (void);

/* ---------------------------------------------------------------- *
 *  Inference.
 * ---------------------------------------------------------------- */

/* Process one FE_FRAME_SIZE-sample frame.
 *   in  : float[FE_FRAME_SIZE]  noisy input
 *   out : float[FE_FRAME_SIZE]  enhanced output (may alias `in`)
 * Maintains streaming state across calls (overlap-add + recurrent caches).
 * The call emits 320 samples; signal alignment includes the fixed 704-sample
 * STFT delay described above. Calling before successful fe_init() is a
 * no-op and leaves `out` untouched. Not reentrant. */
void fe_run  (const float *in, float *out);

/* Reset all streaming state — call between independent audio streams.
 * Calling before successful fe_init() is a no-op. Not thread-safe with
 * fe_run(). */
void fe_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* FE_H */

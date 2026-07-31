// Per-callsite running min/max + sample counter (diagnostics).
// Activation quantization is ASYMMETRIC uint8 stored as int8 = u8 - 128
// (so the existing signedxsigned GEMM kernels run unchanged), recomputed
// per frame from the dynamic max-abs scan. The earlier "freeze after
// warmup" static-int8 chain was removed; this range tracker now only
// records the running range for diagnostics.
#ifndef FE_CALIB_H
#define FE_CALIB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    max_running;
    float    min_running;
    uint32_t samples;
} FeActScale;

static inline void fe_actscale_init(FeActScale *a) {
    a->max_running  = 0.0f;
    a->min_running  = 0.0f;
    a->samples      = 0;
}

/* Result of one asymmetric uint8 quantize pass -- passed by value from the
 * quantize routine to the GEMM wrapper so it can build the bias_eff. */
typedef struct {
    float   scale;
    int32_t zp;       /* uint8 zero-point in [0, 255] */
} FeActQuant;

#ifdef __cplusplus
}
#endif

#endif /* FE_CALIB_H */

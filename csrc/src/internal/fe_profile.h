/*
 * fe_profile.h -- Tiny per-op timing instrumentation.
 *
 * Compile with -DFE_ENABLE_PROFILE to enable. When disabled the macros expand
 * to a single bare expression call so there is exactly zero overhead in
 * release builds.
 *
 * Usage:
 *     FE_TIME("strided_conv", fe_strided_conv1d(...));
 *     FE_TIME("encoder.0",    fe_conv1d_k3_buf_silu(&w->enc[0], cur, nxt,
 *                                                        FE_F1, aq, c32));
 *
 * At process end, call fe_profile_dump(num_frames) to print per-op stats.
 */
#ifndef FE_PROFILE_H
#define FE_PROFILE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef FE_ENABLE_PROFILE

#include <time.h>

double fe_now_us(void);
void   fe_profile_record(const char *name, double us);
void   fe_profile_dump  (int num_frames);
void   fe_profile_reset (void);

/* Time a single expression / statement. Safe inside conditionals. */
#define FE_TIME(name, expr) do { \
        double _t0 = fe_now_us(); \
        (expr); \
        fe_profile_record((name), fe_now_us() - _t0); \
    } while (0)

/* Block-style for multi-statement regions. */
#define FE_TIME_BEGIN(name)  do { const char *_fe_name = (name); double _fe_t0 = fe_now_us();
#define FE_TIME_END()              fe_profile_record(_fe_name, fe_now_us() - _fe_t0); } while (0)

#else  /* FE_ENABLE_PROFILE not set -- zero overhead */

#define FE_TIME(name, expr)   do { (expr); } while (0)
#define FE_TIME_BEGIN(name)   do {
#define FE_TIME_END()         } while (0)

static inline void fe_profile_dump(int num_frames) { (void)num_frames; }
static inline void fe_profile_reset(void)          { }

#endif

#ifdef __cplusplus
}
#endif

#endif /* FE_PROFILE_H */

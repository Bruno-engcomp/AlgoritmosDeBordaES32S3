
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_dsp.h"

#define TAG "BENCH"

#define BENCH_REPS 30
#define WARMUP_REPS 5

#define MAX_FFT_N 4096
#define MAX_FIR_TAPS 256
#define FIR_SAMPLES 4096
#define IIR_SAMPLES 4096

#define CNN_IN 64
#define CNN_KERNEL 3
#define CNN_OUT_CH 8

#define MLP_IN 32
#define MLP_HIDDEN 32
#define MLP_OUT 8

/* Volatile sink prevents the optimizer from removing benchmark work. */
static volatile float g_sink_f = 0.0f;
static volatile int32_t g_sink_i = 0;

typedef void (*bench_fn_t)(void *ctx);

typedef struct {
    uint32_t cycles;
    int64_t us;
    float checksum;
} bench_result_t;

static bench_result_t run_benchmark(bench_fn_t fn, void *ctx)
{
    for (int i = 0; i < WARMUP_REPS; ++i) {
        fn(ctx);
    }

    uint64_t total_cycles = 0;
    int64_t total_us = 0;

    for (int i = 0; i < BENCH_REPS; ++i) {
        uint32_t c0 = esp_cpu_get_cycle_count();
        int64_t t0 = esp_timer_get_time();

        fn(ctx);

        int64_t t1 = esp_timer_get_time();
        uint32_t c1 = esp_cpu_get_cycle_count();

        total_cycles += (uint32_t)(c1 - c0);
        total_us += (t1 - t0);
    }

    bench_result_t r = {
        .cycles = (uint32_t)(total_cycles / BENCH_REPS),
        .us = total_us / BENCH_REPS,
        .checksum = g_sink_f + (float)g_sink_i
    };
    return r;
}

/* ============================================================
 * 1) FFT radix-2/4 float32 — esp-dsp
 * ============================================================ */

typedef struct {
    float *data;
    int n;
} fft_ctx_t;

static void fft_prepare(float *data, int n)
{
    for (int i = 0; i < n; ++i) {
        float x = sinf(2.0f * (float)M_PI * 5.0f * (float)i / (float)n);
        float w = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n);
        data[2*i] = x * w;
        data[2*i + 1] = 0.0f;
    }
}

static void fft_run(void *arg)
{
    fft_ctx_t *ctx = (fft_ctx_t *)arg;

    dsps_fft2r_fc32(ctx->data, ctx->n);
    dsps_bit_rev_fc32(ctx->data, ctx->n);

    /* Read one value so the result is observable. */
    g_sink_f += ctx->data[10] * 1e-7f;

    /* Restore input for the next repetition. */
    fft_prepare(ctx->data, ctx->n);
}

/* ============================================================
 * 2) FIR float32 — esp-dsp
 * ============================================================ */

typedef struct {
    float *input;
    float *output;
    float *coeffs;
    int samples;
    int taps;
    float *state;
} fir_ctx_t;

static void fir_prepare(fir_ctx_t *c)
{
    for (int i = 0; i < c->samples; ++i) {
        c->input[i] = sinf(2.0f * (float)M_PI * 50.0f * i / 1000.0f);
        c->output[i] = 0.0f;
    }

    for (int i = 0; i < c->taps; ++i) {
        c->coeffs[i] = 1.0f / (float)c->taps;
    }

    memset(c->state, 0, sizeof(float) * (c->taps + 4));
}

static void fir_run(void *arg)
{
    fir_ctx_t *c = (fir_ctx_t *)arg;

    dsps_fird_f32(c->input, c->output, c->samples,
                  c->coeffs, c->state, c->taps);

    g_sink_f += c->output[c->samples / 2] * 1e-6f;

    memset(c->state, 0, sizeof(float) * (c->taps + 4));
}

/* ============================================================
 * 3) CNN 1D int8 — reference baseline
 *
 * This is intentionally a small standalone int8 convolution.
 * It establishes a baseline before replacing the kernel with
 * TFLite Micro + ESP-NN.
 * ============================================================ */

typedef struct {
    int8_t input[CNN_IN];
    int8_t kernel[CNN_KERNEL][CNN_OUT_CH];
    int32_t bias[CNN_OUT_CH];
    int8_t output[CNN_IN - CNN_KERNEL + 1][CNN_OUT_CH];
} cnn_ctx_t;

static inline int8_t sat_int8(int32_t x)
{
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static void cnn_run(void *arg)
{
    cnn_ctx_t *c = (cnn_ctx_t *)arg;
    const int out_len = CNN_IN - CNN_KERNEL + 1;

    for (int p = 0; p < out_len; ++p) {
        for (int oc = 0; oc < CNN_OUT_CH; ++oc) {
            int32_t acc = c->bias[oc];

            for (int k = 0; k < CNN_KERNEL; ++k) {
                acc += (int32_t)c->input[p + k] *
                       (int32_t)c->kernel[k][oc];
            }

            /* Simple int8 requantization + ReLU. */
            acc >>= 4;
            if (acc < 0) acc = 0;
            c->output[p][oc] = sat_int8(acc);
        }
    }

    g_sink_i += c->output[10][0];
}

/* ============================================================
 * 4) IIR biquad — esp-dsp
 * ============================================================ */

typedef struct {
    float *input;
    float *output;
    float coeffs[5];
    float *work;
    int samples;
    int sections;
} iir_ctx_t;

static void iir_run(void *arg)
{
    iir_ctx_t *c = (iir_ctx_t *)arg;

    /* For a fair parameter sweep, run multiple biquad sections. */
    for (int s = 0; s < c->sections; ++s) {
        dsps_biquad_f32(c->input, c->output, c->samples,
                        c->coeffs, c->work);

        memcpy(c->input, c->output, sizeof(float) * c->samples);
    }

    g_sink_f += c->output[c->samples / 2] * 1e-6f;

    for (int i = 0; i < c->samples; ++i) {
        c->input[i] = sinf(2.0f * (float)M_PI * 20.0f * i / 1000.0f);
        c->output[i] = 0.0f;
    }
    memset(c->work, 0, sizeof(float) * 2);
}

/* ============================================================
 * 5) MLP dense int8 — reference baseline
 * ============================================================ */

typedef struct {
    int8_t input[MLP_IN];
    int8_t w1[MLP_HIDDEN][MLP_IN];
    int32_t b1[MLP_HIDDEN];

    int8_t hidden[MLP_HIDDEN];

    int8_t w2[MLP_OUT][MLP_HIDDEN];
    int32_t b2[MLP_OUT];

    int8_t output[MLP_OUT];
} mlp_ctx_t;

static void mlp_run(void *arg)
{
    mlp_ctx_t *c = (mlp_ctx_t *)arg;

    for (int h = 0; h < MLP_HIDDEN; ++h) {
        int32_t acc = c->b1[h];

        for (int i = 0; i < MLP_IN; ++i) {
            acc += (int32_t)c->w1[h][i] *
                   (int32_t)c->input[i];
        }

        acc >>= 5;
        if (acc < 0) acc = 0;
        c->hidden[h] = sat_int8(acc);
    }

    for (int o = 0; o < MLP_OUT; ++o) {
        int32_t acc = c->b2[o];

        for (int h = 0; h < MLP_HIDDEN; ++h) {
            acc += (int32_t)c->w2[o][h] *
                   (int32_t)c->hidden[h];
        }

        acc >>= 5;
        c->output[o] = sat_int8(acc);
    }

    g_sink_i += c->output[0];
}

/* ============================================================
 * Initialization
 * ============================================================ */

static void init_cnn(cnn_ctx_t *c)
{
    for (int i = 0; i < CNN_IN; ++i)
        c->input[i] = (int8_t)((i * 7) % 31 - 15);

    for (int k = 0; k < CNN_KERNEL; ++k)
        for (int oc = 0; oc < CNN_OUT_CH; ++oc)
            c->kernel[k][oc] = (int8_t)(((k + 1) * (oc + 2)) % 7 - 3);

    for (int oc = 0; oc < CNN_OUT_CH; ++oc)
        c->bias[oc] = 0;
}

static void init_mlp(mlp_ctx_t *c)
{
    for (int i = 0; i < MLP_IN; ++i)
        c->input[i] = (int8_t)((i * 3) % 17 - 8);

    for (int h = 0; h < MLP_HIDDEN; ++h) {
        c->b1[h] = 0;
        for (int i = 0; i < MLP_IN; ++i)
            c->w1[h][i] = (int8_t)(((h + i) % 9) - 4);
    }

    for (int o = 0; o < MLP_OUT; ++o) {
        c->b2[o] = 0;
        for (int h = 0; h < MLP_HIDDEN; ++h)
            c->w2[o][h] = (int8_t)(((o + h) % 7) - 3);
    }
}

static void print_result(const char *name, int parameter,
                         bench_result_t r, float checksum)
{
    printf("CSV,%s,%d,%lu,%lld,%.6f\n",
           name,
           parameter,
           (unsigned long)r.cycles,
           (long long)r.us,
           checksum);
}

/* ============================================================
 * Main benchmark
 * ============================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3 benchmark: first 5 algorithms");
    ESP_LOGI(TAG, "CPU cycle counter + esp_timer");
    ESP_LOGI(TAG, "Repetitions=%d warmup=%d", BENCH_REPS, WARMUP_REPS);

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, MAX_FFT_N);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %d", ret);
        return;
    }

    /* --------------------------------------------------------
     * FFT: N = 64 ... 4096
     * -------------------------------------------------------- */
    float *fft_data = heap_caps_aligned_alloc(
        16, sizeof(float) * MAX_FFT_N * 2, MALLOC_CAP_8BIT);

    if (!fft_data) {
        ESP_LOGE(TAG, "Cannot allocate FFT buffer");
        return;
    }

    for (int n = 64; n <= MAX_FFT_N; n *= 2) {
        fft_ctx_t c = {
            .data = fft_data,
            .n = n
        };

        fft_prepare(c.data, c.n);
        bench_result_t r = run_benchmark(fft_run, &c);

        print_result("FFT", n, r, c.data[0]);
    }

    free(fft_data);

    /* --------------------------------------------------------
     * FIR: taps = 8 ... 256
     * -------------------------------------------------------- */
    float *fir_input = heap_caps_malloc(sizeof(float) * FIR_SAMPLES,
                                        MALLOC_CAP_8BIT);
    float *fir_output = heap_caps_malloc(sizeof(float) * FIR_SAMPLES,
                                         MALLOC_CAP_8BIT);
    float *fir_coeffs = heap_caps_malloc(sizeof(float) * MAX_FIR_TAPS,
                                         MALLOC_CAP_8BIT);
    float *fir_state = heap_caps_malloc(sizeof(float) * (MAX_FIR_TAPS + 4),
                                        MALLOC_CAP_8BIT);

    if (!fir_input || !fir_output || !fir_coeffs || !fir_state) {
        ESP_LOGE(TAG, "Cannot allocate FIR buffers");
        return;
    }

    for (int taps = 8; taps <= MAX_FIR_TAPS; taps *= 2) {
        fir_ctx_t c = {
            .input = fir_input,
            .output = fir_output,
            .coeffs = fir_coeffs,
            .samples = FIR_SAMPLES,
            .taps = taps,
            .state = fir_state
        };

        fir_prepare(&c);
        bench_result_t r = run_benchmark(fir_run, &c);

        /* cycles/sample is more useful for the report. */
        r.cycles /= FIR_SAMPLES;
        r.us = (r.us * 1000) / FIR_SAMPLES;

        print_result("FIR", taps, r, c.output[FIR_SAMPLES / 2]);
    }

    free(fir_input);
    free(fir_output);
    free(fir_coeffs);
    free(fir_state);

    /* --------------------------------------------------------
     * CNN 1D int8 baseline
     * -------------------------------------------------------- */
    cnn_ctx_t cnn;
    memset(&cnn, 0, sizeof(cnn));
    init_cnn(&cnn);

    bench_result_t cnn_r = run_benchmark(cnn_run, &cnn);
    float cnn_ms = (float)cnn_r.us / 1000.0f;
    float cnn_fps = (cnn_ms > 0.0f) ? 1000.0f / cnn_ms : 0.0f;

    printf("CNN_RESULT,ms_per_inference=%.3f,FPS=%.2f,cycles=%lu\n",
           cnn_ms, cnn_fps, (unsigned long)cnn_r.cycles);

    print_result("CNN", CNN_OUT_CH, cnn_r,
                 (float)cnn.output[10][0]);

    /* --------------------------------------------------------
     * IIR: 2 ... 8 sections
     * -------------------------------------------------------- */
    float *iir_input = heap_caps_malloc(sizeof(float) * IIR_SAMPLES,
                                        MALLOC_CAP_8BIT);
    float *iir_output = heap_caps_malloc(sizeof(float) * IIR_SAMPLES,
                                         MALLOC_CAP_8BIT);
    float *iir_work = heap_caps_calloc(2, sizeof(float), MALLOC_CAP_8BIT);

    if (!iir_input || !iir_output || !iir_work) {
        ESP_LOGE(TAG, "Cannot allocate IIR buffers");
        return;
    }

    iir_ctx_t iir = {
        .input = iir_input,
        .output = iir_output,
        .samples = IIR_SAMPLES,
        .work = iir_work
    };

    /* Low-pass biquad coefficients generated by esp-dsp. */
    ret = dsps_biquad_gen_lpf_f32(iir.coeffs, 0.10f, 1.0f);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Could not generate IIR coefficients: %d", ret);
        return;
    }

    for (int sections = 2; sections <= 8; sections += 2) {
        iir.sections = sections;

        for (int i = 0; i < IIR_SAMPLES; ++i)
            iir.input[i] = sinf(2.0f * (float)M_PI * 20.0f * i / 1000.0f);

        memset(iir.output, 0, sizeof(float) * IIR_SAMPLES);
        memset(iir.work, 0, sizeof(float) * 2);

        bench_result_t r = run_benchmark(iir_run, &iir);

        r.cycles /= IIR_SAMPLES;
        r.us = (r.us * 1000) / IIR_SAMPLES;

        print_result("IIR", sections, r, iir.output[IIR_SAMPLES / 2]);
    }

    free(iir_input);
    free(iir_output);
    free(iir_work);

    /* --------------------------------------------------------
     * MLP int8 baseline
     * -------------------------------------------------------- */
    mlp_ctx_t mlp;
    memset(&mlp, 0, sizeof(mlp));
    init_mlp(&mlp);

    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    bench_result_t mlp_r = run_benchmark(mlp_run, &mlp);
    size_t free_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    size_t activation_bytes =
        sizeof(mlp.hidden) + sizeof(mlp.output);

    printf("MLP_RESULT,ms_per_inference=%.3f,cycles=%lu,"
           "activation_bytes=%u,free_internal_delta=%u\n",
           (float)mlp_r.us / 1000.0f,
           (unsigned long)mlp_r.cycles,
           (unsigned)activation_bytes,
           (unsigned)(free_before - free_after));

    print_result("MLP", MLP_HIDDEN, mlp_r,
                 (float)mlp.output[0]);

    ESP_LOGI(TAG, "Benchmark finished. sink_f=%f sink_i=%ld",
             (double)g_sink_f, (long)g_sink_i);

    dsps_fft2r_deinit_fc32();
}

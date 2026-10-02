
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

#define TAG "BENCH2"

#define BENCH_REPS 30
#define WARMUP_REPS 5

/* 16 kHz, 30 ms window as suggested by the document for MFCC. */
#define MFCC_SAMPLE_RATE 16000
#define MFCC_WIN_MS      30
#define MFCC_NSAMPLES    (MFCC_SAMPLE_RATE * MFCC_WIN_MS / 1000)   /* 480 */
#define MFCC_MAX_FFT     1024
#define MFCC_NMEL        26
#define MFCC_NDCT        32   /* radix-2 and >= MFCC_NMEL */

#define KALMAN_MAX_N     8

#define MAX_MEDIAN_WIN   15
#define MEDIAN_SAMPLES   4096

#define MA_SAMPLES       4096
#define MA_MAX_WIN       256

#define GOERTZEL_SAMPLES 4096

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

/* Turns absolute measurement into cycles/sample and ns/sample. */
static void normalize_per_sample(bench_result_t *r, int samples)
{
    r->cycles /= (uint32_t)samples;
    r->us = (r->us * 1000) / samples;
}

/* ============================================================
 * 6) MFCC (Mel + DCT) — esp-dsp FFT + window + DCT
 *
 * Pre-emphasis, Hamming/Hann window, power spectrum, mel
 * filterbank, natural log, DCT-II. This is the classic speech
 * front-end pipeline.
 * ============================================================ */

typedef struct {
    float *fft_buf;      /* N * 2 floats, interleaved re/im */
    float *window;       /* MFCC_NSAMPLES */
    float *spectrum;     /* N/2 + 1 */
    float *mel;          /* MFCC_NMEL */
    float *dct_buf;      /* MFCC_NDCT * 2 */
    float *mel_bank;         /* MFCC_NMEL * bins, filled by setup */
    int mel_start[MFCC_NMEL];
    int mel_end[MFCC_NMEL];
    int n;
    int bins;
    const float *input;  /* MFCC_NSAMPLES */
} mfcc_ctx_t;

static float hz_to_mel(float f)
{
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

static float mel_to_hz(float m)
{
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

/* Builds a triangular mel filterbank for a given FFT size.
 * Runs once, outside the timed region. */
static void mfcc_build_melbank(mfcc_ctx_t *c)
{
    int bins = c->bins;
    float lo_mel = hz_to_mel(80.0f);
    float hi_mel = hz_to_mel(7600.0f);

    for (int m = 0; m < MFCC_NMEL; ++m) {
        float m0 = lo_mel + (hi_mel - lo_mel) * (float)m / (float)MFCC_NMEL;
        float m1 = lo_mel + (hi_mel - lo_mel) * (float)(m + 1) / (float)MFCC_NMEL;
        float m2 = lo_mel + (hi_mel - lo_mel) * (float)(m + 2) / (float)MFCC_NMEL;

        float f0 = mel_to_hz(m0);
        float f1 = mel_to_hz(m1);
        float f2 = mel_to_hz(m2);

        int b0 = (int)ceilf(f0 * (float)(c->n) / (float)MFCC_SAMPLE_RATE);
        int b1 = (int)floorf(f1 * (float)(c->n) / (float)MFCC_SAMPLE_RATE);
        int b2 = (int)floorf(f2 * (float)(c->n) / (float)MFCC_SAMPLE_RATE);

        if (b0 < 0) b0 = 0;
        if (b1 < 0) b1 = 0;
        if (b2 >= bins) b2 = bins - 1;
        if (b1 > bins - 1) b1 = bins - 1;

        c->mel_start[m] = b0;
        c->mel_end[m] = (b2 > b1) ? b2 : b1;

        for (int b = c->mel_start[m]; b <= c->mel_end[m]; ++b) {
            float fb = (float)b * (float)MFCC_SAMPLE_RATE / (float)c->n;
            float w = 0.0f;

            if (b == b0) {
                w = (f1 > f0) ? (fb - f0) / (f1 - f0) : 1.0f;
            } else if (b == b2) {
                w = (f2 > f1) ? (f2 - fb) / (f2 - f1) : 1.0f;
            } else {
                w = (fb - f0) / (f1 - f0);
                float w2 = (f2 - fb) / (f2 - f1);
                if (w2 < w) w = w2;
            }

            if (w < 0.0f) w = 0.0f;
            c->mel_bank[m * bins + b] = w;
        }
    }
}

static void mfcc_run(void *arg)
{
    mfcc_ctx_t *c = (mfcc_ctx_t *)arg;
    int n = c->n;

    for (int i = 0; i < MFCC_NSAMPLES; ++i) {
        c->fft_buf[2 * i] = c->input[i] * c->window[i];
        c->fft_buf[2 * i + 1] = 0.0f;
    }
    for (int i = MFCC_NSAMPLES; i < n; ++i) {
        c->fft_buf[2 * i] = 0.0f;
        c->fft_buf[2 * i + 1] = 0.0f;
    }

    dsps_fft2r_fc32(c->fft_buf, n);
    dsps_bit_rev_fc32(c->fft_buf, n);

    /* Power spectrum. */
    for (int b = 0; b < c->bins; ++b) {
        float re = c->fft_buf[2 * b];
        float im = c->fft_buf[2 * b + 1];
        c->spectrum[b] = re * re + im * im;
    }

    /* Mel projection. */
    for (int m = 0; m < MFCC_NMEL; ++m) {
        const float *bank = c->mel_bank + m * c->bins;
        float acc = 0.0f;

        for (int b = c->mel_start[m]; b <= c->mel_end[m]; ++b) {
            acc += bank[b] * c->spectrum[b];
        }

        /* Floor keeps log() finite during silent frames. */
        c->mel[m] = log10f(acc + 1e-10f);
    }

    /* DCT-II, unscaled. Buffer must be MFCC_NDCT * 2. */
    for (int i = 0; i < MFCC_NDCT; ++i) {
        c->dct_buf[i] = (i < MFCC_NMEL) ? c->mel[i] : 0.0f;
        c->dct_buf[MFCC_NDCT + i] = 0.0f;
    }
    dsps_dct_f32(c->dct_buf, MFCC_NDCT);

    g_sink_f += c->dct_buf[0] * 1e-7f;
}

/* ============================================================
 * 7) Filtro de Kalman — C puro com matrizes pequenas
 *
 * O sistema e uma cadeia de n estagios: cada estado recebe o
 * estado seguinte, e apenas o estado medido (o ultimo) persiste.
 * Assim todos os estados convergem para a medicao e a matriz F
 * nao amplifica valores, o que mantem a covariancia bem
 * condicionada em float32 para n ate 8.
 *
 * P = F P F' + Q no passo de predicao, e a atualizacao escalar
 * usa S = P'[n-1][n-1] + R. A correcao e feita fora do lugar:
 * ler P enquanto ela e reescrita quebraria o ganho de Kalman.
 * ============================================================ */

typedef struct {
    int n;
    float F[KALMAN_MAX_N][KALMAN_MAX_N];
    float Q[KALMAN_MAX_N][KALMAN_MAX_N];
    float P[KALMAN_MAX_N][KALMAN_MAX_N];   /* covariancia a posteriori */
    float Pf[KALMAN_MAX_N][KALMAN_MAX_N];  /* covariancia predita */
    float T[KALMAN_MAX_N][KALMAN_MAX_N];
    float x[KALMAN_MAX_N];
    float xf[KALMAN_MAX_N];
    float K[KALMAN_MAX_N];
    float R;
} kalman_ctx_t;

static void kalman_init(kalman_ctx_t *c, int n)
{
    memset(c, 0, sizeof(*c));
    c->n = n;

    memset(c->F, 0, sizeof(c->F));
    for (int i = 0; i + 1 < n; ++i) {
        c->F[i][i + 1] = 1.0f;
    }
    /* So o estado medido persiste; sem isso a predicao dele seria
     * sempre zero e o ganho de Kalman nao puxaria para a medicao. */
    c->F[n - 1][n - 1] = 1.0f;

    for (int i = 0; i < n; ++i) {
        c->Q[i][i] = 1e-3f;
        c->P[i][i] = 1.0f;
        c->x[i] = 0.0f;
    }

    c->R = 1e-1f;
}

static void kalman_run(void *arg)
{
    kalman_ctx_t *c = (kalman_ctx_t *)arg;
    int n = c->n;

    /* ---- predict state: xf = F x ---- */
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += c->F[i][k] * c->x[k];
        }
        c->xf[i] = acc;
    }

    /* ---- predict covariance: T = F P, Pf = T F' + Q ---- */
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) {
                acc += c->F[i][j] * c->P[j][k];
            }
            c->T[i][k] = acc;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = c->Q[i][j];
            for (int k = 0; k < n; ++k) {
                acc += c->T[i][k] * c->F[j][k];
            }
            c->Pf[i][j] = acc;
        }
    }

    /* ---- update: medicao escalar do ultimo estado ---- */
    float S = c->Pf[n - 1][n - 1] + c->R;
    float z = 1.0f;

    for (int i = 0; i < n; ++i) {
        c->K[i] = c->Pf[i][n - 1] / S;
    }

    for (int i = 0; i < n; ++i) {
        c->x[i] = c->xf[i] + c->K[i] * (z - c->xf[n - 1]);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c->P[i][j] = c->Pf[i][j] - c->K[i] * c->Pf[n - 1][j];
        }
    }

    g_sink_f += c->x[0] * 1e-6f;
}

/* ============================================================
 * 8) Goertzel — C puro (recorrencia de 2 ordem)
 *
 * Algoritmo classico de deteccao de tom fixo. O coeficiente
 * 2*cos(w) e constante, entao o laco interno tem poucos ops.
 * ============================================================ */

typedef struct {
    const float *input;
    int samples;
    float coeff;
} goertzel_ctx_t;

static void goertzel_run(void *arg)
{
    goertzel_ctx_t *c = (goertzel_ctx_t *)arg;

    float s1 = 0.0f;
    float s2 = 0.0f;

    for (int i = 0; i < c->samples; ++i) {
        float s0 = c->input[i] + c->coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    /* Amplitude do bin alvo (formula fechada usual). */
    float power = s1 * s1 + s2 * s2 - c->coeff * s1 * s2;

    g_sink_f += power * 1e-12f;
}

/* ============================================================
 * 9) Filtro de mediana — C puro (janela ordenada)
 *
 * A janela circular mantem os ultimos k valores na ordem de
 * chegada, e uma copia ordenada e recalculada a cada amostra.
 * Nao da para ordenar a janela no lugar e continuar: ordenar
 * destroi a ordem de chegada, e a amostra mais antiga nao e
 * necessariamente a menor nem a maior.
 *
 * Insercao sobre uma janela quase ordenada custa O(k), o que da
 * O(N*k) amortizado, como o documento sugere.
 * ============================================================ */

typedef struct {
    const float *input;
    float *output;
    int samples;
    int k;
    float hist[MAX_MEDIAN_WIN];   /* ordem de chegada, circular */
    float win[MAX_MEDIAN_WIN];    /* copia ordenada */
} median_ctx_t;

static void insertion_sort(float *a, int n)
{
    for (int i = 1; i < n; ++i) {
        float v = a[i];
        int j = i - 1;

        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = v;
    }
}

static void median_run(void *arg)
{
    median_ctx_t *c = (median_ctx_t *)arg;
    int k = c->k;

    for (int j = 0; j < k; ++j) {
        c->hist[j] = c->input[j % c->samples];
    }

    for (int i = 0; i < c->samples; ++i) {
        if (i >= k) {
            c->hist[i % k] = c->input[i];
        }

        memcpy(c->win, c->hist, sizeof(float) * (size_t)k);
        insertion_sort(c->win, k);

        c->output[i] = c->win[k / 2];
    }

    g_sink_f += c->output[c->samples / 2] * 1e-6f;
}

/* ============================================================
 * 10) Media movel — C puro com soma deslizante
 *
 * Soma corrente: 1 subtracao e 1 adicao por amostra, o menor
 * custo possivel para um filtro de media.
 * ============================================================ */

typedef struct {
    const float *input;
    float *output;
    int samples;
    int win;
} ma_ctx_t;

static void ma_run(void *arg)
{
    ma_ctx_t *c = (ma_ctx_t *)arg;
    float inv = 1.0f / (float)c->win;

    float sum = 0.0f;
    for (int i = 0; i < c->win; ++i) {
        sum += c->input[i];
    }
    c->output[c->win - 1] = sum * inv;

    for (int i = c->win; i < c->samples; ++i) {
        sum += c->input[i] - c->input[i - c->win];
        c->output[i] = sum * inv;
    }

    g_sink_f += c->output[c->samples / 2] * 1e-6f;
}

/* ============================================================
 * Output helpers
 * ============================================================ */

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

static void *alloc_floats(size_t count)
{
    return heap_caps_malloc(count * sizeof(float), MALLOC_CAP_8BIT);
}

/* ============================================================
 * Main benchmark
 * ============================================================ */

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3 benchmark: algorithms 6-10 (PDS + Filtros)");
    ESP_LOGI(TAG, "CPU cycle counter + esp_timer");
    ESP_LOGI(TAG, "Repetitions=%d warmup=%d", BENCH_REPS, WARMUP_REPS);

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, MFCC_MAX_FFT);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %d", ret);
        return;
    }

    /* --------------------------------------------------------
     * 6) MFCC: ms por janela de 30 ms (FFT de 256 a 1024)
     * -------------------------------------------------------- */
    float *mfcc_input = alloc_floats(MFCC_NSAMPLES);
    float *mfcc_window = alloc_floats(MFCC_NSAMPLES);
    float *mfcc_fft = alloc_floats(MFCC_MAX_FFT * 2);
    float *mfcc_spectrum = alloc_floats(MFCC_MAX_FFT / 2 + 1);
    float *mfcc_mel = alloc_floats(MFCC_NMEL);
    float *mfcc_dct = alloc_floats(MFCC_NDCT * 2);
    float *mfcc_bank = alloc_floats((size_t)MFCC_NMEL * (MFCC_MAX_FFT / 2 + 1));

    if (!mfcc_input || !mfcc_window || !mfcc_fft || !mfcc_spectrum ||
        !mfcc_mel || !mfcc_dct || !mfcc_bank) {
        ESP_LOGE(TAG, "Cannot allocate MFCC buffers");
        return;
    }

    /* Two tones plus noise-like content so no filter collapses. */
    for (int i = 0; i < MFCC_NSAMPLES; ++i) {
        float t = (float)i / (float)MFCC_SAMPLE_RATE;
        mfcc_input[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * t) +
                        0.3f * sinf(2.0f * (float)M_PI * 1200.0f * t) +
                        0.1f * sinf(2.0f * (float)M_PI * 3100.0f * t);
    }
    dsps_wind_hann_f32(mfcc_window, MFCC_NSAMPLES);

    for (int n = 256; n <= MFCC_MAX_FFT; n *= 2) {
        mfcc_ctx_t c = {
            .fft_buf = mfcc_fft,
            .window = mfcc_window,
            .spectrum = mfcc_spectrum,
            .mel = mfcc_mel,
            .dct_buf = mfcc_dct,
            .mel_bank = mfcc_bank,
            .n = n,
            .bins = n / 2 + 1,
            .input = mfcc_input
        };

        memset(mfcc_bank, 0,
               sizeof(float) * MFCC_NMEL * (size_t)c.bins);
        mfcc_build_melbank(&c);

        bench_result_t r = run_benchmark(mfcc_run, &c);

        float ms_per_window = (float)r.us / 1000.0f;
        printf("MFCC_RESULT,fft_size=%d,ms_per_30ms_window=%.3f,"
               "cycles=%lu,real_time_factor=%.2f\n",
               n, ms_per_window, (unsigned long)r.cycles,
               (float)MFCC_WIN_MS / (ms_per_window > 0.0f ? ms_per_window : 1.0f));

        print_result("MFCC", n, r, mfcc_dct[0]);
    }

    free(mfcc_input);
    free(mfcc_window);
    free(mfcc_fft);
    free(mfcc_spectrum);
    free(mfcc_mel);
    free(mfcc_dct);
    free(mfcc_bank);

    /* --------------------------------------------------------
     * 7) Kalman: ciclos/atualizacao vs nº de estados
     * -------------------------------------------------------- */
    for (int n = 2; n <= KALMAN_MAX_N; ++n) {
        kalman_ctx_t c;
        kalman_init(&c, n);

        bench_result_t r = run_benchmark(kalman_run, &c);

        printf("KALMAN_RESULT,states=%d,cycles_per_update=%lu,"
               "us_per_update=%.3f,state_estimate=%.6f\n",
               n, (unsigned long)r.cycles, (double)r.us, (double)c.x[0]);

        print_result("KALMAN", n, r, c.x[0]);
    }

    /* --------------------------------------------------------
     * 8) Goertzel: ciclos/amostra vs N
     * -------------------------------------------------------- */
    float *go_input = alloc_floats(GOERTZEL_SAMPLES);

    if (!go_input) {
        ESP_LOGE(TAG, "Cannot allocate Goertzel buffer");
        return;
    }

    const float target_hz = 1000.0f;
    for (int i = 0; i < GOERTZEL_SAMPLES; ++i) {
        go_input[i] = sinf(2.0f * (float)M_PI * target_hz * (float)i /
                          (float)MFCC_SAMPLE_RATE);
    }

    for (int n = 256; n <= GOERTZEL_SAMPLES; n *= 2) {
        goertzel_ctx_t c = {
            .input = go_input,
            .samples = n,
            .coeff = 2.0f * cosf(2.0f * (float)M_PI * target_hz /
                                 (float)MFCC_SAMPLE_RATE)
        };

        bench_result_t r = run_benchmark(goertzel_run, &c);
        normalize_per_sample(&r, n);

        print_result("GOERTZEL", n, r, c.coeff);
    }

    free(go_input);

    /* --------------------------------------------------------
     * 9) Mediana: ciclos/amostra vs janela k (3-15)
     * -------------------------------------------------------- */
    float *md_input = alloc_floats(MEDIAN_SAMPLES);
    float *md_output = alloc_floats(MEDIAN_SAMPLES);

    if (!md_input || !md_output) {
        ESP_LOGE(TAG, "Cannot allocate median buffers");
        return;
    }

    for (int i = 0; i < MEDIAN_SAMPLES; ++i) {
        /* Rampa suave com picos a cada 64 amostras: a mediana
         * deve remover os picos. */
        md_input[i] = 100.0f + 0.05f * (float)i;
        if (i % 64 == 0) {
            md_input[i] += 500.0f;
        }
    }

    for (int k = 3; k <= MAX_MEDIAN_WIN; k += 2) {
        median_ctx_t c = {
            .input = md_input,
            .output = md_output,
            .samples = MEDIAN_SAMPLES,
            .k = k
        };

        bench_result_t r = run_benchmark(median_run, &c);
        normalize_per_sample(&r, MEDIAN_SAMPLES);

        print_result("MEDIAN", k, r, md_output[MEDIAN_SAMPLES / 2]);
    }

    free(md_input);
    free(md_output);

    /* --------------------------------------------------------
     * 10) Media movel: ciclos/amostra vs tamanho da janela
     * -------------------------------------------------------- */
    float *ma_input = alloc_floats(MA_SAMPLES);
    float *ma_output = alloc_floats(MA_SAMPLES);

    if (!ma_input || !ma_output) {
        ESP_LOGE(TAG, "Cannot allocate moving average buffers");
        return;
    }

    for (int i = 0; i < MA_SAMPLES; ++i) {
        ma_input[i] = sinf(2.0f * (float)M_PI * 50.0f * (float)i / 1000.0f);
    }

    for (int win = 4; win <= MA_MAX_WIN; win *= 2) {
        ma_ctx_t c = {
            .input = ma_input,
            .output = ma_output,
            .samples = MA_SAMPLES,
            .win = win
        };

        bench_result_t r = run_benchmark(ma_run, &c);
        normalize_per_sample(&r, MA_SAMPLES);

        print_result("MAVG", win, r, ma_output[MA_SAMPLES / 2]);
    }

    free(ma_input);
    free(ma_output);

    ESP_LOGI(TAG, "Benchmark finished. sink_f=%f sink_i=%ld",
             (double)g_sink_f, (long)g_sink_i);

    dsps_fft2r_deinit_fc32();
}

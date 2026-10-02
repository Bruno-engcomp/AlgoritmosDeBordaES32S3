/*
 * Benchmark ESP32-S3 - algoritmos 11 a 16 do documento
 * "Algoritmos de borda no ESP32 - S3 prioridades para benchmark".
 *
 * Continuacao de esp32s3_benchmark_5_algoritmos (itens 1 a 5) e
 * esp32s3_benchmark_5_algoritmos_2 (itens 6 a 10).
 *
 *   11) LMS adaptativo     - C puro (laco adaptativo)   O(N*M)
 *   12) SVM (linear/RBF)   - C puro (laco sobre SV)     O(D*SV)
 *   13) Random Forest      - C puro (arvore if-else)    O(T*P)
 *   14) k-NN               - C puro (dist. euclidiana)  O(N*D)
 *   15) DWT (Haar / db4)   - C puro (banco de filtros)  O(N)
 *   16) DTW                - C puro (prog. dinamica)    O(N*M)
 *
 * Diferenca em relacao aos projetos anteriores: nenhum destes seis
 * itens usa FFT nem banco de filtros do esp-dsp, entao a dependencia
 * espressif/esp-dsp foi omitida de proposito. Todos sao C puro.
 *
 * O documento cita micromlgen e emlearn para SVM e Random Forest.
 * Nenhuma das duas esta disponivel como componente do ESP-IDF (o
 * emlearn e Arduino/C++), e o padrao dos projetos anteriores e
 * implementar em C puro para ficar autocontido e comparavel. Ver
 * a secao "Por que em C puro" no README.
 */

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

#define TAG "BENCH3"

#define BENCH_REPS 30
#define WARMUP_REPS 5

/* ---- 11) LMS adaptativo ---- */
#define LMS_SAMPLES     4096
#define LMS_MAX_TAPS    128
#define LMS_SYS_TAPS    8      /* sistema desconhecido a identificar */
#define LMS_MU          0.5f

/* ---- 12) SVM ---- */
#define SVM_DIM         13     /* dimensoes de vetor, tipico de MFCC */
#define SVM_MAX_SV      128
#define SVM_QUERIES     32
#define SVM_GAMMA       (1.0f / (float)SVM_DIM)

/* ---- 13) Random Forest ---- */
#define RF_DIM          13
#define RF_TRAIN        256
#define RF_QUERIES      32
#define RF_MAX_TREES    64
#define RF_MAX_DEPTH    5
#define RF_NCLASS       2
/* Arvore de profundidade D tem no maximo 2^(D+1)-1 nos; D=5 => 63.
 * Cada arvore ocupa uma fatia de RF_MAX_NODES para que os nos de
 * uma arvore fiquem contiguos e a inferencia possa andar por
 * deslocamento fixo. */
#define RF_MAX_NODES    ((1 << (RF_MAX_DEPTH + 1)) - 1)

/* ---- 14) k-NN ---- */
#define KNN_DIM         13
#define KNN_MAX_DB      1024
#define KNN_QUERIES     32
#define KNN_K           3

/* ---- 15) DWT ---- */
#define DWT_SAMPLES     4096
#define DWT_MAX_LEVELS  6

/* ---- 16) DTW ---- */
#define DTW_DIM         8
#define DTW_LEN         128
#define DTW_MAX_BAND    (DTW_LEN - 1)

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
 * Gerador deterministico
 *
 * LCG de 32 bits. Cada conjunto de dados e regerado a partir de
 * uma semente fixa antes do uso, para que os resultados nao
 * dependam da ordem em que os parametros sao varridos.
 * ============================================================ */

static uint32_t g_rng = 1u;

static void rng_seed(uint32_t s)
{
    g_rng = s ? s : 1u;
}

static uint32_t rng_next(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

/* Uniforme em [0,1). Usa os 24 bits de cima porque os bits baixos
 * do LCG variam pouco e perderiam precisao ao virar float32. */
static float rng_uni01(void)
{
    return (float)(rng_next() >> 8) * (1.0f / 16777216.0f);
}

static float rng_bipolar(void)
{
    return 2.0f * rng_uni01() - 1.0f;
}

/* Soma de 4 uniformes aproxima uma gaussiana sem usar log nem cos,
 * o que manteria o custo do gerador longe do laco medido. */
static float rng_gauss(void)
{
    return rng_bipolar() + rng_bipolar() + rng_bipolar() + rng_bipolar();
}

static void insertion_sort(float *a, int n)
{
    for (int i = 1; i < n; ++i) {
        const float v = a[i];
        int j = i - 1;

        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = v;
    }
}

/* ============================================================
 * 11) LMS adaptativo - C puro (laco adaptativo)
 *
 * NLMS (LMS normalizado). A forma classica do LMS usa um passo
 * fixo mu, e em float32 ela diverge assim que mu passa de
 * 2/(M*var(x)): com M grande e mu fixo o peso estoura. O NLMS
 * divide pela potencia instantanea do vetor de entrada, o que
 * deixa o passo efetivo sempre em (0,2) independente de M, e e
 * por isso que mu e constante entre todos os M varridos.
 *
 * O filtro e causal: xi = &x[t] traz a janela das ultimas M
 * amostras com a mais recente em xi[M-1]. A referencia lida e
 * d[t+M-1], o mesmo instante. Por isso os coeficientes
 * convergem para h invertido no tempo, w[m] = h[M-1-m].
 *
 * A janela e acessada por ponteiro, sem buffer circular nem
 * memmove: e a forma mais rapida, porque o laco interno tem
 * varredura contigua e o compilador vetoriza.
 * ============================================================ */

typedef struct {
    const float *x;      /* entrada: ruido branco */
    const float *d;      /* referencia: saida do sistema desconhecido */
    float *w;            /* LMS_MAX_TAPS */
    float sum_e2;        /* energia de erro acumulada na execucao */
    int samples;
    int taps;
} lms_ctx_t;

static void nlms_run(void *arg)
{
    lms_ctx_t *c = (lms_ctx_t *)arg;
    const int taps = c->taps;
    const int n = c->samples - taps + 1;
    const int t0 = taps - 1;

    /* Recomeca do zero a cada execucao: o que se mede e o custo de
     * adaptar N amostras desde o inicio, nao de um filtro ja
     * adaptado, senao o peso deixa de se mexer e a conta encolhe. */
    memset(c->w, 0, sizeof(float) * (size_t)taps);

    float sum_e2 = 0.0f;

    for (int t = 0; t < n; ++t) {
        const float *xi = &c->x[t];
        const float ref = c->d[t + t0];

        float y = 0.0f;
        float power = 0.0f;
        for (int m = 0; m < taps; ++m) {
            y += c->w[m] * xi[m];
            power += xi[m] * xi[m];
        }

        const float e = ref - y;
        sum_e2 += e * e;

        /* power tem piso para o NLMS nao dividir por zero no
         * transiente em que a janela ainda esta quase zerada. */
        const float k = (LMS_MU / (power + 1e-6f)) * e;

        for (int m = 0; m < taps; ++m) {
            c->w[m] += k * xi[m];
        }
    }

    c->sum_e2 = sum_e2;
    g_sink_f += c->w[0] * 1e-6f;
}

/* Passagem NAO cronometrada: conta em quantas amostras a potencia
 * media do erro cai 20 dB em relacao ao instante inicial. */
static int lms_iters_to_20db(const lms_ctx_t *m)
{
    float w[LMS_MAX_TAPS];
    const int taps = m->taps;
    const int n = m->samples - taps + 1;
    const int t0 = taps - 1;

    memset(w, 0, sizeof(float) * (size_t)taps);

    float e0 = 0.0f;
    for (int t = 0; t < n; ++t) {
        const float *xi = &m->x[t];
        float y = 0.0f;
        for (int k = 0; k < taps; ++k) {
            y += w[k] * xi[k];
        }
        const float e = m->d[t + t0] - y;
        e0 += e * e;
    }

    if (e0 <= 0.0f) {
        return 0;
    }
    const float target = e0 * 0.01f;   /* -20 dB em potencia */

    float acc = 0.0f;
    for (int t = 0; t < n; ++t) {
        const float *xi = &m->x[t];
        float y = 0.0f;
        float power = 0.0f;
        for (int k = 0; k < taps; ++k) {
            y += w[k] * xi[k];
            power += xi[k] * xi[k];
        }

        const float e = m->d[t + t0] - y;
        acc += e * e;

        const float k = (LMS_MU / (power + 1e-6f)) * e;
        for (int j = 0; j < taps; ++j) {
            w[j] += k * xi[j];
        }

        if ((t & 63) == 63) {
            const float win = 64.0f * acc / (float)(t + 1);
            if (win <= target) {
                return t + 1;
            }
        }
    }

    return -1;   /* nao convergiu dentro da janela */
}

/* ============================================================
 * 12) SVM - C puro (laco sobre vetores-suporte)
 *
 * Avalia a funcao de decisao na forma dual,
 *
 *     f(x) = soma_i  alfa_i * y_i * K(x, sv_i) + b
 *
 * que e a forma que o micromlgen e o emlearn geram em C: um laco
 * sobre os vetores-suporte. O custo e O(D*SV) por inferencia, e o
 * numero de vetores-suporte e o parametro que decide se o modelo
 * cabe na MCU.
 *
 * Para o kernel linear a soma pode ser reescrita como w.x + b com
 * w = soma alfa_i*y_i*sv_i, caindo para O(D) e dispensando a base
 * de treino inteira em RAM. Isso e conferido em app_main: a forma
 * dual e a reduzida tem de dar o mesmo resultado, e a diferenca de
 * tempo entre as duas e o preco de nao fazer essa otimizacao.
 * ============================================================ */

typedef struct {
    const float *sv;     /* n_sv * SVM_DIM */
    const float *alpha;  /* n_sv, ja com o sinal do rotulo */
    const float *query;  /* SVM_DIM */
    float w[SVM_DIM];    /* forma reduzida do kernel linear */
    float b;
    float out;
    int n_sv;
    int rbf;             /* 0 = linear, 1 = RBF */
} svm_ctx_t;

static float svm_decide(svm_ctx_t *c)
{
    const float *q = c->query;
    float acc = c->b;

    for (int i = 0; i < c->n_sv; ++i) {
        const float *s = &c->sv[(size_t)i * SVM_DIM];
        const float a = c->alpha[i];

        if (c->rbf) {
            /* Distancia ao quadrado: a raiz e monotonica e nao muda
             * a ordem, entao nao e preciso extrair. O gargalo real
             * do RBF e o exp(), nao a distancia. */
            float d2 = 0.0f;
            for (int j = 0; j < SVM_DIM; ++j) {
                const float t = s[j] - q[j];
                d2 += t * t;
            }
            acc += a * expf(-SVM_GAMMA * d2);
        } else {
            float dot = 0.0f;
            for (int j = 0; j < SVM_DIM; ++j) {
                dot += s[j] * q[j];
            }
            acc += a * dot;
        }
    }

    c->out = acc;
    return acc;
}

static void svm_run(void *arg)
{
    svm_ctx_t *c = (svm_ctx_t *)arg;
    svm_decide(c);
    g_sink_f += c->out * 1e-6f;
}

/* Forma reduzida do kernel linear: O(SVM_DIM) em vez de O(D*SV). */
static float svm_decide_collapsed(const svm_ctx_t *c)
{
    float acc = c->b;
    for (int j = 0; j < SVM_DIM; ++j) {
        acc += c->w[j] * c->query[j];
    }
    return acc;
}

/* ============================================================
 * 13) Random Forest - C puro (if-else)
 *
 * A floresta e um array plano de nos, percorrido com if-else e um
 * indice de proximo no. E a forma que o emlearn gera: sem ponteiros
 * e sem alocacao por no, e o custo e previsivel porque a
 * profundidade limita o numero de desvios.
 *
 * O treino e de verdade aqui: bootstrap com reposicao, busca do
 * melhor limiar por Gini sobre um subconjunto aleatorio de
 * atributos (mtry), tudo na montagem, fora da regiao medida.
 *
 * Os indices das amostras de um no ficam em buffers estaticos
 * indexados pela profundidade: aloca-los na pilha custaria 2 KB por
 * nivel e a task principal do FreeRTOS tem pilha curta.
 * ============================================================ */

typedef struct {
    int feature;     /* -1 = folha */
    float threshold;
    int left;
    int right;
    int class_id;
} rf_node_t;

typedef struct {
    rf_node_t *nodes;
    int tree_size[RF_MAX_TREES];   /* nos usados por arvore */
    int n_trees;
    int depth;
} rf_forest_t;

/* scratch de particionamento, um par de vetores por profundidade */
static int rf_split[RF_MAX_DEPTH + 1][2][RF_TRAIN];
static float rf_vals[RF_TRAIN];

static int rf_label(const float *row)
{
    const int c = (int)(row[0] + 0.5f);
    return (c < 0) ? 0 : ((c >= RF_NCLASS) ? RF_NCLASS - 1 : c);
}

static int rf_add_node(rf_forest_t *f, int tree, int idx)
{
    rf_node_t *nd = &f->nodes[(size_t)tree * RF_MAX_NODES + (size_t)idx];

    nd->feature = -1;
    nd->threshold = 0.0f;
    nd->left = -1;
    nd->right = -1;
    nd->class_id = 0;

    return idx;
}

/* Contagem de classes do conjunto indicado por idx[0..n). */
static void rf_count(const float *train, const int *idx, int n, int *cnt)
{
    cnt[0] = 0;
    cnt[1] = 0;
    for (int i = 0; i < n; ++i) {
        cnt[rf_label(&train[(size_t)idx[i] * RF_DIM])]++;
    }
}

static void rf_grow(rf_forest_t *f, int tree, int node, int depth,
                    const float *train, const int *idx, int n)
{
    int cnt[RF_NCLASS];
    rf_count(train, idx, n, cnt);

    const int majority = (cnt[1] > cnt[0]) ? 1 : 0;

    /* Folha: profundidade maxima, amostra insuficiente ou no puro. */
    if (depth >= f->depth || n < 2 || cnt[0] == 0 || cnt[1] == 0) {
        f->nodes[(size_t)tree * RF_MAX_NODES + (size_t)node].class_id = majority;
        return;
    }

    const int mtry = 1 + (int)sqrtf((float)RF_DIM);
    const float total = (float)(n * cnt[1] * (n - cnt[1]));

    int best_feat = -1;
    float best_thr = 0.0f;
    float best_impurity = total + 1.0f;

    for (int t = 0; t < mtry; ++t) {
        const int feat = (int)(rng_next() % (uint32_t)RF_DIM);

        /* Limiares candidatos: pontos medios entre valores
         * distintos e consecutivos do proprio no. */
        for (int i = 0; i < n; ++i) {
            rf_vals[i] = train[(size_t)idx[i] * RF_DIM + feat];
        }
        insertion_sort(rf_vals, n);

        for (int i = 0; i + 1 < n; ++i) {
            if (rf_vals[i] == rf_vals[i + 1]) {
                continue;   /* limiar entre valores iguais nao separa */
            }
            const float thr = 0.5f * (rf_vals[i] + rf_vals[i + 1]);

            int lc[RF_NCLASS] = { 0, 0 };
            int rc[RF_NCLASS] = { 0, 0 };

            for (int k = 0; k < n; ++k) {
                const float *row = &train[(size_t)idx[k] * RF_DIM];
                if (row[feat] <= thr) {
                    lc[rf_label(row)]++;
                } else {
                    rc[rf_label(row)]++;
                }
            }

            const int nl = lc[0] + lc[1];
            const int nr = rc[0] + rc[1];
            if (nl == 0 || nr == 0) {
                continue;
            }

            /* Gini ponderado pelo tamanho do lado. */
            const float gl = 1.0f - ((float)(lc[0] * lc[0]) +
                                     (float)(lc[1] * lc[1])) / ((float)nl * (float)nl);
            const float gr = 1.0f - ((float)(rc[0] * rc[0]) +
                                     (float)(rc[1] * rc[1])) / ((float)nr * (float)nr);
            const float impurity = gl * (float)nl + gr * (float)nr;

            if (impurity < best_impurity) {
                best_impurity = impurity;
                best_feat = feat;
                best_thr = thr;
            }
        }
    }

    if (best_feat < 0) {
        f->nodes[(size_t)tree * RF_MAX_NODES + (size_t)node].class_id = majority;
        return;
    }

    int *left_idx = rf_split[depth][0];
    int *right_idx = rf_split[depth][1];
    int nl = 0;
    int nr = 0;

    for (int i = 0; i < n; ++i) {
        const int s = idx[i];
        if (train[(size_t)s * RF_DIM + best_feat] <= best_thr) {
            left_idx[nl++] = s;
        } else {
            right_idx[nr++] = s;
        }
    }

    if (nl == 0 || nr == 0) {
        f->nodes[(size_t)tree * RF_MAX_NODES + (size_t)node].class_id = majority;
        return;
    }

    const int lnode = rf_add_node(f, tree, f->tree_size[tree]++);
    const int rnode = rf_add_node(f, tree, f->tree_size[tree]++);

    rf_node_t *nd = &f->nodes[(size_t)tree * RF_MAX_NODES + (size_t)node];
    nd->feature = best_feat;
    nd->threshold = best_thr;
    nd->left = lnode;
    nd->right = rnode;
    nd->class_id = -1;

    rf_grow(f, tree, lnode, depth + 1, train, left_idx, nl);
    rf_grow(f, tree, rnode, depth + 1, train, right_idx, nr);
}

static void rf_build(rf_forest_t *f, const float *train, int n_trees)
{
    f->n_trees = n_trees;
    f->depth = RF_MAX_DEPTH;
    memset(f->tree_size, 0, sizeof(f->tree_size));

    /* Ordem deterministica: a arvore t comeca sempre no mesmo
     * ponto do gerador, entao a floresta de 8 arvores e prefixo da
     * de 64 e o efeito do numero de arvores fica isolado. */
    rng_seed(0xF022u);

    for (int t = 0; t < n_trees; ++t) {
        rf_add_node(f, t, f->tree_size[t]++);

        int bag[RF_TRAIN];
        for (int i = 0; i < RF_TRAIN; ++i) {
            bag[i] = (int)(rng_next() % (uint32_t)RF_TRAIN);
        }

        rf_grow(f, t, 0, 0, train, bag, RF_TRAIN);
    }
}

/* Inferencia: if-else puro, como o codigo gerado pelo emlearn. */
static int rf_tree_predict(const rf_node_t *nodes, const float *x)
{
    int i = 0;
    while (nodes[i].feature >= 0) {
        const rf_node_t *nd = &nodes[i];
        i = (x[nd->feature] <= nd->threshold) ? nd->left : nd->right;
    }
    return nodes[i].class_id;
}

typedef struct {
    const rf_forest_t *forest;
    const float *query;      /* RF_DIM */
    int out;
} rf_ctx_t;

static void rf_run(void *arg)
{
    rf_ctx_t *c = (rf_ctx_t *)arg;
    const rf_forest_t *f = c->forest;

    int votes[RF_NCLASS] = { 0, 0 };

    for (int t = 0; t < f->n_trees; ++t) {
        const rf_node_t *base = &f->nodes[(size_t)t * RF_MAX_NODES];
        votes[rf_tree_predict(base, c->query)]++;
    }

    c->out = (votes[1] > votes[0]) ? 1 : 0;
    g_sink_i += c->out;
}

/* ============================================================
 * 14) k-NN - C puro (distancia euclidiana)
 *
 * Distancia ao quadrado, sem raiz: a raiz e monotonica e nao muda
 * a ordem das distancias, entao o conjunto de vizinhos e o mesmo.
 * Omite-se N raizes por inferencia.
 *
 * e o mais caro da lista em tempo: o custo cresce linearmente com
 * a base, e a base precisa residir na SRAM. E por isso que a
 * metrica do documento e ms/inferencia contra o tamanho da base.
 * ============================================================ */

typedef struct {
    const float *base;      /* n_db * KNN_DIM */
    const float *query;     /* KNN_DIM */
    float *best;            /* KNN_K indices */
    float *best_d;          /* KNN_K distancias */
    int n_db;
    int k;
    int out;
} knn_ctx_t;

static void knn_run(void *arg)
{
    knn_ctx_t *c = (knn_ctx_t *)arg;
    const int k = c->k;

    for (int i = 0; i < k; ++i) {
        c->best[i] = -1.0f;
        c->best_d[i] = 1e30f;
    }

    for (int i = 0; i < c->n_db; ++i) {
        const float *b = &c->base[(size_t)i * KNN_DIM];
        float d2 = 0.0f;

        for (int j = 0; j < KNN_DIM - 1; ++j) {
            const float t = b[j] - c->query[j];
            d2 += t * t;
        }

        /* Descarta antes de escrever: se nem entra no top-k, nao ha
         * o que atualizar. Numa base grande a maioria cai aqui. */
        if (d2 >= c->best_d[k - 1]) {
            continue;
        }

        /* Insercao ordenada em K posicoes: O(K), nao O(n). */
        int pos = k - 1;
        while (pos > 0 && c->best_d[pos - 1] > d2) {
            c->best_d[pos] = c->best_d[pos - 1];
            c->best[pos] = c->best[pos - 1];
            pos--;
        }
        c->best_d[pos] = d2;
        c->best[pos] = (float)i;
    }

    /* Voto por maioria dos K vizinhos. */
    int votes = 0;
    for (int i = 0; i < k; ++i) {
        if (c->best[i] >= 0.0f &&
            c->base[(size_t)c->best[i] * KNN_DIM + KNN_DIM - 1] > 0.5f) {
            votes++;
        }
    }
    c->out = (votes * 2 > k) ? 1 : 0;

    g_sink_f += c->best_d[0] * 1e-6f;
}

/* ============================================================
 * 15) DWT (Haar / db4) - C puro (banco de filtros)
 *
 * Transformada wavelet discreta por nivel, em banco de filtros
 * ortonormal: passa-baixa seguido de subamostragem por 2, e o mesmo
 * no passa-alta. O detalhe e a aproximacao; a aproximacao e levada
 * ao proximo nivel, entao o custo por nivel cai pela metade e a
 * soma da progressao geometrica e O(N).
 *
 * Os filtros low/high sao um par QMF de energia unitaria,
 * high[j] = -(-1)^j * low[L-1-j]. Consequencia pratica: cada nivel
 * preserva a energia da propria entrada,
 *
 *     soma(a_l^2) + soma(d_l^2) = soma(x_l^2)
 *
 * e o energy_ratio = 1.000000 impresso aqui e a verificacao
 * numerica, valendo para Haar e para db4.
 *
 * AoSomar os niveis, a energia e contada uma vez so: guardam-se
 * todos os detalhes d_0..d_{L-1} e apenas a aproximacao final
 * a_{L-1}. Somar tambem a_0, a_1, ... contaria a mesma energia
 * varias vezes, porque a_l e a entrada do nivel l+1.
 *
 * O indice e circular, k = (2*i + j) mod n. Como 2*i < n e j < L,
 * uma unica subtracao resolve o wrap: nao ha modulo no laco
 * interno, que e o laco quente.
 * ============================================================ */

static const float dwt_lo_haar[2] = {
    0.7071067811865475f, 0.7071067811865475f
};
static const float dwt_hi_haar[2] = {
    -0.7071067811865475f, 0.7071067811865475f
};

/* Daubechies 4. low e ortonormal; high e o par QMF correspondente. */
static const float dwt_lo_db4[4] = {
    0.4829629131445341f,
    0.8365163037378079f,
    0.2241438680420134f,
    -0.1294095225512604f
};
static const float dwt_hi_db4[4] = {
    0.1294095225512604f,
    0.2241438680420134f,
    -0.8365163037378079f,
    0.4829629131445341f
};

typedef struct {
    const float *in;            /* DWT_SAMPLES */
    const float *lo;
    const float *hi;
    float *a[DWT_MAX_LEVELS];   /* ponteiros dentro de arena */
    float *d[DWT_MAX_LEVELS];
    int taps;
    int samples;
    int levels;
} dwt_ctx_t;

static void dwt_one_level(const float *in, int n,
                          const float *lo, const float *hi, int L,
                          float *a, float *d)
{
    const int m = n / 2;

    for (int i = 0; i < m; ++i) {
        float sa = 0.0f;
        float sd = 0.0f;
        const int base = 2 * i;

        for (int j = 0; j < L; ++j) {
            int k = base + j;
            if (k >= n) {
                k -= n;    /* wrap circular, um passo so */
            }
            sa += lo[j] * in[k];
            sd += hi[j] * in[k];
        }

        a[i] = sa;
        d[i] = sd;
    }
}

static void dwt_run(void *arg)
{
    dwt_ctx_t *c = (dwt_ctx_t *)arg;

    const float *cur = c->in;
    int n = c->samples;

    for (int lvl = 0; lvl < c->levels; ++lvl) {
        dwt_one_level(cur, n, c->lo, c->hi, c->taps, c->a[lvl], c->d[lvl]);
        cur = c->a[lvl];
        n /= 2;
    }

    g_sink_f += c->d[0][0] * 1e-6f;
}

/* ============================================================
 * 16) DTW - C puro (programacao dinamica)
 *
 * D[i][j] = c(i,j) + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
 *
 * Sem restricao, a matriz inteira custa O(N*M) floats: com N=M=128
 * sao 64 KB so nela, e e o que torna o DTW classico caro demais
 * para borda.
 *
 * A janela Sakoe-Chiba limita |i - j| <= r. Como cada linha so toca
 * 2r+1 colunas, duas linhas bastam: a memoria cai de O(N*M) para
 * O(2r+1), que e a metrica de RAM que o documento pede.
 *
 * Mapeamento coluna -> posicao na linha: p = j - i + r + 1. Como p
 * desloca exatamente 1 a cada linha, duas linhas de 2r+3 floats
 * bastam e nenhuma delas precisa ser redimensionada. Os slots 0 e
 * 2r+2 sao as guardas (fora da janela): ficam em infinito e nunca
 * sao escritos nem lidos pelo laco.
 *
 * D[i-1][j]    -> prev[p + 1]
 * D[i-1][j-1]  -> prev[p]
 * D[i][j-1]    -> cur[p - 1]
 * ============================================================ */

#define DTW_INF 1e30f

typedef struct {
    const float *a;        /* DTW_LEN * DTW_DIM */
    const float *b;        /* DTW_LEN * DTW_DIM */
    float *row0;
    float *row1;
    int len;
    int band;              /* r da janela Sakoe-Chiba */
    int slots;             /* 2r + 3 */
    float dist;
} dtw_ctx_t;

static void dtw_run(void *arg)
{
    dtw_ctx_t *c = (dtw_ctx_t *)arg;
    const int n = c->len;
    const int r = c->band;
    const int slots = c->slots;

    float *prev = c->row0;
    float *cur = c->row1;

    /* A linha virtual -1 e a coluna virtual -1 valem infinito: e o
     * que obriga o caminho a comecar em (0,0). Sem esta
     * inicializacao, prev comeca com lixo de heap e a primeira
     * linha aceita partir de um custo invalido, o que abaixa a
     * distancia sem motivo. */
    for (int t = 0; t < slots; ++t) {
        prev[t] = DTW_INF;
        cur[t] = DTW_INF;
    }

    for (int i = 0; i < n; ++i) {
        const float *ai = &c->a[(size_t)i * DTW_DIM];

        /* Faixa valida da linha i: j em [i-r, i+r] limitado a [0,n-1].
         * A borda em j=0 e i=0 existe (e a propagacao ao longo dela e
         * correta no DTW padrao); o infinito fica na linha -1. */
        const int jlo = (i - r > 0) ? (i - r) : 0;
        const int jhi = (i + r < n - 1) ? (i + r) : (n - 1);

        /* A linha vira toda infinito, incluindo as guardas. Sem
         * isso, os slots que a janela de cima nao alcanca ficariam
         * com valor velho de duas linhas atras e aceitariam um
         * caminho que a janela proibe. Custa O(2r+1) por linha,
         * mesma ordem do laco de custo. */
        for (int t = 0; t < slots; ++t) {
            cur[t] = DTW_INF;
        }

        for (int j = jlo; j <= jhi; ++j) {
            const float *bj = &c->b[(size_t)j * DTW_DIM];

            float d2 = 0.0f;
            for (int k = 0; k < DTW_DIM; ++k) {
                const float t = ai[k] - bj[k];
                d2 += t * t;
            }

            const int p = j - i + r + 1;
            float best;

            if (i == 0 && j == 0) {
                best = 0.0f;
            } else {
                best = prev[p + 1];
                if (prev[p] < best) {
                    best = prev[p];
                }
                if (cur[p - 1] < best) {
                    best = cur[p - 1];
                }
            }

            cur[p] = best + d2;
        }

        float *tmp = prev;
        prev = cur;
        cur = tmp;
    }

    /* Celula final (n-1, n-1): p = (n-1) - (n-1) + r + 1 = r + 1. */
    c->dist = prev[r + 1];
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
    ESP_LOGI(TAG, "ESP32-S3 benchmark: algoritmos 11-16");
    ESP_LOGI(TAG, "Filtro adaptativo, classificadores e PDS, todos em C puro");
    ESP_LOGI(TAG, "Repetitions=%d warmup=%d", BENCH_REPS, WARMUP_REPS);

    /* --------------------------------------------------------
     * 11) LMS: ciclos/amostra vs taps + taxa de convergencia
     * -------------------------------------------------------- */
    {
        float *x = alloc_floats(LMS_SAMPLES);
        float *d = alloc_floats(LMS_SAMPLES);
        float *w = alloc_floats(LMS_MAX_TAPS);

        if (!x || !d || !w) {
            ESP_LOGE(TAG, "Cannot allocate LMS buffers");
            return;
        }

        /* Sistema desconhecido de 8 taps. A referencia e a saida
         * desse sistema para o mesmo ruido, entao o LMS tem o que
         * adaptar sem nunca ver os coeficientes verdadeiros. */
        float sys_h[LMS_SYS_TAPS];
        rng_seed(0x51A5u);
        for (int i = 0; i < LMS_SYS_TAPS; ++i) {
            sys_h[i] = rng_gauss() * 0.35f;
        }

        rng_seed(0xBEEF01u);
        for (int i = 0; i < LMS_SAMPLES; ++i) {
            x[i] = rng_gauss();      /* ruido branco, variancia 1 */
        }

        for (int n = 0; n < LMS_SAMPLES; ++n) {
            float acc = 0.0f;
            for (int m = 0; m < LMS_SYS_TAPS; ++m) {
                if (n - m >= 0) {
                    acc += sys_h[m] * x[n - m];
                }
            }
            d[n] = acc;
        }

        float d_energy = 0.0f;
        for (int i = 0; i < LMS_SAMPLES; ++i) {
            d_energy += d[i] * d[i];
        }

        for (int taps = 8; taps <= LMS_MAX_TAPS; taps *= 2) {
            lms_ctx_t c = {
                .x = x,
                .d = d,
                .w = w,
                .samples = LMS_SAMPLES,
                .taps = taps
            };

            bench_result_t r = run_benchmark(nlms_run, &c);

            const int eff = LMS_SAMPLES - taps + 1;
            normalize_per_sample(&r, eff);

            /* Erro residual medio em relacao a referencia. Com
             * excitacao sem ruido de medicao o NLMS converge ate a
             * precisao da maquina, entao este numero e dominado
             * pelo transiente de convergencia e nao por ruido de
             * gradiente em regime. E a taxa de convergencia que
             * fica no campo iters_to_20db. */
            const float e_err = c.sum_e2 / (float)eff;
            const float e_ref = d_energy / (float)LMS_SAMPLES;
            const float res_db = (e_ref > 0.0f)
                               ? 10.0f * log10f((e_err / e_ref) + 1e-30f)
                               : 0.0f;

            /* w e o tempo invertido de h: w[m] = h[taps-1-m]. */
            float werr = 0.0f;
            for (int m = 0; m < LMS_SYS_TAPS; ++m) {
                const float t = w[taps - 1 - m] - sys_h[m];
                werr += t * t;
            }

            printf("LMS_RESULT,taps=%d,cycles_per_sample=%lu,ns_per_sample=%lld,"
                   "residual_mse_db=%.3f,iters_to_20db=%d,weight_rmse=%.3e\n",
                   taps, (unsigned long)r.cycles, (long long)r.us,
                   (double)res_db, lms_iters_to_20db(&c),
                   (double)sqrtf(werr / (float)LMS_SYS_TAPS));

            print_result("LMS", taps, r, w[0]);
        }

        free(x);
        free(d);
        free(w);
    }

    /* --------------------------------------------------------
     * 12) SVM: micros por inferencia vs nº de vetores-suporte
     * -------------------------------------------------------- */
    {
        float *sv = alloc_floats((size_t)SVM_MAX_SV * SVM_DIM);
        float *alpha = alloc_floats(SVM_MAX_SV);
        float *queries = alloc_floats((size_t)SVM_QUERIES * SVM_DIM);

        if (!sv || !alpha || !queries) {
            ESP_LOGE(TAG, "Cannot allocate SVM buffers");
            return;
        }

        /* Dois blocos bem separados em -0.6 e +0.6 mais ruido: o
         * modelo e trivialmente separavel, o que mantem a acuracia
         * alta e isola o custo da inferencia do custo de um modelo
         * ruim. */
        rng_seed(0x5A17u);
        for (int i = 0; i < SVM_MAX_SV; ++i) {
            const int cls = i & 1;
            const float center = cls ? 0.6f : -0.6f;

            for (int j = 0; j < SVM_DIM; ++j) {
                sv[(size_t)i * SVM_DIM + j] = center + rng_gauss() * 0.35f;
            }
            /* alfa ja com o sinal do rotulo */
            alpha[i] = (cls ? 0.5f : -0.5f) / (float)SVM_MAX_SV;
        }

        rng_seed(0x5A18u);
        for (int q = 0; q < SVM_QUERIES; ++q) {
            const int cls = q & 1;
            const float center = cls ? 0.6f : -0.6f;

            for (int j = 0; j < SVM_DIM; ++j) {
                queries[(size_t)q * SVM_DIM + j] = center + rng_gauss() * 0.35f;
            }
        }

        for (int kernel = 0; kernel < 2; ++kernel) {
            const char *kname = kernel ? "SVMRBF" : "SVM";

            for (int n_sv = 8; n_sv <= SVM_MAX_SV; n_sv *= 2) {
                /* w = soma alfa_i * y_i * sv_i; so o kernel linear
                 * admite essa reescrita. */
                float w[SVM_DIM];
                memset(w, 0, sizeof(w));
                if (!kernel) {
                    for (int i = 0; i < n_sv; ++i) {
                        for (int j = 0; j < SVM_DIM; ++j) {
                            w[j] += alpha[i] * sv[(size_t)i * SVM_DIM + j];
                        }
                    }
                }

                float max_diff = 0.0f;
                double us_sum = 0.0;
                double cyc_sum = 0.0;
                int correct = 0;

                for (int q = 0; q < SVM_QUERIES; ++q) {
                    svm_ctx_t c = {
                        .sv = sv,
                        .alpha = alpha,
                        .query = &queries[(size_t)q * SVM_DIM],
                        .n_sv = n_sv,
                        .rbf = kernel,
                        .b = 0.0f,
                        .out = 0.0f
                    };
                    memcpy(c.w, w, sizeof(w));

                    bench_result_t r = run_benchmark(svm_run, &c);

                    if (!kernel) {
                        const float diff =
                            fabsf(c.out - svm_decide_collapsed(&c));
                        if (diff > max_diff) {
                            max_diff = diff;
                        }
                    }

                    if (((c.out >= 0.0f) ? 1 : 0) == (q & 1)) {
                        correct++;
                    }

                    us_sum += (double)r.us;
                    cyc_sum += (double)r.cycles;
                }

                /* Custo da forma reduzida, so para o kernel linear:
                 * e o preco de nao colapsar os vetores-suporte. */
                double us_collapsed = 0.0;
                if (!kernel) {
                    for (int rep = 0; rep < BENCH_REPS; ++rep) {
                        for (int q = 0; q < SVM_QUERIES; ++q) {
                            svm_ctx_t c = {
                                .sv = sv,
                                .alpha = alpha,
                                .query = &queries[(size_t)q * SVM_DIM],
                                .n_sv = n_sv,
                                .rbf = 0,
                                .b = 0.0f,
                                .out = 0.0f
                            };
                            memcpy(c.w, w, sizeof(w));

                            const int64_t t0 = esp_timer_get_time();
                            c.out = svm_decide_collapsed(&c);
                            us_collapsed += (double)(esp_timer_get_time() - t0);
                        }
                    }
                    us_collapsed /= (double)(BENCH_REPS * SVM_QUERIES);
                }

                const float us_avg = (float)(us_sum / (double)SVM_QUERIES);
                const float acc = (float)correct / (float)SVM_QUERIES;

                bench_result_t r = {
                    .cycles = (uint32_t)(cyc_sum / (double)SVM_QUERIES),
                    .us = (int64_t)us_avg,
                    .checksum = 0.0f
                };

                if (kernel) {
                    printf("SVMRBF_RESULT,n_sv=%d,us_per_inference=%.3f,"
                           "cycles_per_inference=%lu,gamma=%.6f,accuracy=%.3f\n",
                           n_sv, (double)us_avg, (unsigned long)r.cycles,
                           (double)SVM_GAMMA, (double)acc);
                } else {
                    printf("SVM_RESULT,n_sv=%d,us_per_inference=%.3f,"
                           "cycles_per_inference=%lu,us_collapsed=%.3f,"
                           "dual_vs_collapsed_diff=%.3e,accuracy=%.3f\n",
                           n_sv, (double)us_avg, (unsigned long)r.cycles,
                           us_collapsed, (double)max_diff, (double)acc);
                }

                print_result(kname, n_sv, r, us_avg);
            }
        }

        free(sv);
        free(alpha);
        free(queries);
    }

    /* --------------------------------------------------------
     * 13) Random Forest: micros por inferencia e flash vs arvores
     * -------------------------------------------------------- */
    {
        float *train = alloc_floats((size_t)RF_TRAIN * RF_DIM);

        if (!train) {
            ESP_LOGE(TAG, "Cannot allocate RF training buffer");
            return;
        }

        /* Classe na dimensao 0 (0.0 ou 1.0), atributos nas demais. */
        rng_seed(0xF022u);
        for (int i = 0; i < RF_TRAIN; ++i) {
            const int cls = i & 1;
            const float center = cls ? 0.5f : -0.5f;
            train[(size_t)i * RF_DIM + 0] = (float)cls;

            for (int j = 1; j < RF_DIM; ++j) {
                train[(size_t)i * RF_DIM + j] = center + rng_gauss() * 0.4f;
            }
        }

        for (int trees = 8; trees <= RF_MAX_TREES; trees *= 2) {
            rf_node_t *nodes = heap_caps_malloc(
                (size_t)trees * RF_MAX_NODES * sizeof(rf_node_t),
                MALLOC_CAP_8BIT);
            if (!nodes) {
                ESP_LOGE(TAG, "RF: not enough memory for %d trees", trees);
                break;
            }

            rf_forest_t forest = { .nodes = nodes, .n_trees = trees,
                                   .depth = RF_MAX_DEPTH };

            rf_build(&forest, train, trees);

            int used = 0;
            for (int t = 0; t < trees; ++t) {
                used += forest.tree_size[t];
            }
            /* Custo de memoria do modelo, que e o que o documento
             * chama de flash: em geracao de codigo os nos vao para
             * a flash como if-else. */
            const size_t model_bytes = (size_t)used * sizeof(rf_node_t);

            double us_sum = 0.0;
            double cyc_sum = 0.0;
            int correct = 0;

            for (int q = 0; q < RF_QUERIES; ++q) {
                const int cls = q & 1;
                const float center = cls ? 0.5f : -0.5f;
                float query[RF_DIM];

                rng_seed(0xA1CE0000u + (uint32_t)q);
                for (int j = 0; j < RF_DIM; ++j) {
                    query[j] = (j == 0) ? (float)cls
                                        : (center + rng_gauss() * 0.4f);
                }

                rf_ctx_t c = { .forest = &forest, .query = query, .out = 0 };
                bench_result_t r = run_benchmark(rf_run, &c);

                if (c.out == cls) {
                    correct++;
                }
                us_sum += (double)r.us;
                cyc_sum += (double)r.cycles;
            }

            const float us_avg = (float)(us_sum / (double)RF_QUERIES);
            const float acc = (float)correct / (float)RF_QUERIES;

            bench_result_t r = {
                .cycles = (uint32_t)(cyc_sum / (double)RF_QUERIES),
                .us = (int64_t)us_avg,
                .checksum = acc
            };

            printf("RFOREST_RESULT,trees=%d,us_per_inference=%.3f,"
                   "cycles_per_inference=%lu,nodes=%d,nodes_per_tree=%d,"
                   "model_bytes=%lu,model_kib=%.1f,depth=%d,accuracy=%.3f,"
                   "min_free_heap=%lu\n",
                   trees, (double)us_avg, (unsigned long)r.cycles, used,
                   used / trees, (unsigned long)model_bytes,
                   (double)(model_bytes / 1024.0), forest.depth, (double)acc,
                   (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT));

            print_result("RFOREST", trees, r, acc);

            free(nodes);
        }

        free(train);
    }

    /* --------------------------------------------------------
     * 14) k-NN: micros por inferencia vs tamanho da base
     * -------------------------------------------------------- */
    {
        float *base = alloc_floats((size_t)KNN_MAX_DB * KNN_DIM);
        float *queries = alloc_floats((size_t)KNN_QUERIES * KNN_DIM);
        float *best = alloc_floats(KNN_K);
        float *best_d = alloc_floats(KNN_K);

        if (!base || !queries || !best || !best_d) {
            ESP_LOGE(TAG, "Cannot allocate kNN buffers");
            return;
        }

        /* Rotulo na ultima dimensao, atributos nas demais. */
        rng_seed(0xC0FFEEu);
        for (int i = 0; i < KNN_MAX_DB; ++i) {
            const int cls = i & 1;
            const float center = cls ? 0.5f : -0.5f;
            base[(size_t)i * KNN_DIM + KNN_DIM - 1] = (float)cls;

            for (int j = 0; j < KNN_DIM - 1; ++j) {
                base[(size_t)i * KNN_DIM + j] = center + rng_gauss() * 0.4f;
            }
        }

        rng_seed(0xC0FFEFu);
        for (int q = 0; q < KNN_QUERIES; ++q) {
            const int cls = q & 1;
            const float center = cls ? 0.5f : -0.5f;
            queries[(size_t)q * KNN_DIM + KNN_DIM - 1] = (float)cls;

            for (int j = 0; j < KNN_DIM - 1; ++j) {
                queries[(size_t)q * KNN_DIM + j] = center + rng_gauss() * 0.4f;
            }
        }

        for (int n_db = 64; n_db <= KNN_MAX_DB; n_db *= 2) {
            double us_sum = 0.0;
            double cyc_sum = 0.0;
            int correct = 0;

            for (int q = 0; q < KNN_QUERIES; ++q) {
                knn_ctx_t c = {
                    .base = base,
                    .query = &queries[(size_t)q * KNN_DIM],
                    .best = best,
                    .best_d = best_d,
                    .n_db = n_db,
                    .k = KNN_K,
                    .out = 0
                };

                bench_result_t r = run_benchmark(knn_run, &c);

                if (c.out == (q & 1)) {
                    correct++;
                }
                us_sum += (double)r.us;
                cyc_sum += (double)r.cycles;
            }

            const float us_avg = (float)(us_sum / (double)KNN_QUERIES);
            const float acc = (float)correct / (float)KNN_QUERIES;
            const size_t base_bytes = (size_t)n_db * KNN_DIM * sizeof(float);

            bench_result_t r = {
                .cycles = (uint32_t)(cyc_sum / (double)KNN_QUERIES),
                .us = (int64_t)us_avg,
                .checksum = best_d[0]
            };

            printf("KNN_RESULT,db_size=%d,k=%d,us_per_inference=%.3f,"
                   "cycles_per_inference=%lu,accuracy=%.3f,base_bytes=%lu,"
                   "base_kib=%.1f\n",
                   n_db, KNN_K, (double)us_avg, (unsigned long)r.cycles,
                   (double)acc, (unsigned long)base_bytes,
                   (double)(base_bytes / 1024.0));

            print_result("KNN", n_db, r, best_d[0]);

            (void)r;
        }

        free(base);
        free(queries);
        free(best);
        free(best_d);
    }

    /* --------------------------------------------------------
     * 15) DWT: ciclos vs numero de niveis (Haar e db4)
     * -------------------------------------------------------- */
    {
        float *in = alloc_floats(DWT_SAMPLES);

        /* Cada nivel usa metade do anterior, entao o total e a soma
         * geometrica, nao 6 vezes a metade cheia. */
        size_t arena_floats = 0;
        for (int l = 0; l < DWT_MAX_LEVELS; ++l) {
            arena_floats += 2u * (size_t)(DWT_SAMPLES >> (l + 1));
        }
        float *arena = alloc_floats(arena_floats);

        if (!in || !arena) {
            ESP_LOGE(TAG, "Cannot allocate DWT buffers");
            free(in);
            free(arena);
            return;
        }

        rng_seed(0xD07u);
        for (int i = 0; i < DWT_SAMPLES; ++i) {
            in[i] = rng_gauss();
        }

        float ein = 0.0f;
        for (int i = 0; i < DWT_SAMPLES; ++i) {
            ein += in[i] * in[i];
        }

        for (int wavelet = 0; wavelet < 2; ++wavelet) {
            const char *wname = wavelet ? "DWT_DB4" : "DWT_HAAR";
            const float *lo = wavelet ? dwt_lo_db4 : dwt_lo_haar;
            const float *hi = wavelet ? dwt_hi_db4 : dwt_hi_haar;
            const int L = wavelet ? 4 : 2;

            for (int levels = 1; levels <= DWT_MAX_LEVELS; ++levels) {
                dwt_ctx_t c = {
                    .in = in,
                    .lo = lo,
                    .hi = hi,
                    .taps = L,
                    .samples = DWT_SAMPLES,
                    .levels = levels
                };

                /* Layout calculado uma vez, fora da regiao medida. */
                size_t off = 0;
                for (int l = 0; l < DWT_MAX_LEVELS; ++l) {
                    const size_t sz = (size_t)(DWT_SAMPLES >> (l + 1));
                    c.a[l] = arena + off;
                    off += sz;
                    c.d[l] = arena + off;
                    off += sz;
                }

                bench_result_t r = run_benchmark(dwt_run, &c);

                /* Parseval: cada nivel preserva a energia da sua
                 * entrada. A energia total e a soma de todos os
                 * detalhes mais a aproximacao final: somar tambem
                 * as aproximacoes intermediarias contaria a mesma
                 * energia mais de uma vez, ja que a_l e entrada do
                 * nivel l+1. */
                float eout = 0.0f;

                for (int l = 0; l < levels; ++l) {
                    const int m = DWT_SAMPLES >> (l + 1);
                    for (int i = 0; i < m; ++i) {
                        eout += c.d[l][i] * c.d[l][i];
                    }
                }

                {
                    const int lf = levels - 1;
                    const int mf = DWT_SAMPLES >> (lf + 1);
                    for (int i = 0; i < mf; ++i) {
                        eout += c.a[lf][i] * c.a[lf][i];
                    }
                }

                const float eratio = (ein > 0.0f) ? (eout / ein) : 0.0f;

                printf("%s_RESULT,levels=%d,samples=%d,cycles=%lu,"
                       "cycles_per_sample=%.2f,energy_ratio=%.6f\n",
                       wname, levels, DWT_SAMPLES, (unsigned long)r.cycles,
                       (double)((float)r.cycles / (float)DWT_SAMPLES),
                       (double)eratio);

                print_result(wname, levels, r, eratio);
            }
        }

        free(in);
        free(arena);
    }

    /* --------------------------------------------------------
     * 16) DTW: micros por comparacao e RAM vs janela Sakoe-Chiba
     * -------------------------------------------------------- */
    {
        const int max_slots = 2 * DTW_MAX_BAND + 3;

        float *a = alloc_floats((size_t)DTW_LEN * DTW_DIM);
        float *b = alloc_floats((size_t)DTW_LEN * DTW_DIM);
        float *row0 = alloc_floats((size_t)max_slots);
        float *row1 = alloc_floats((size_t)max_slots);

        if (!a || !b || !row0 || !row1) {
            ESP_LOGE(TAG, "Cannot allocate DTW buffers");
            return;
        }

        rng_seed(0xD7A1u);
        for (int i = 0; i < DTW_LEN; ++i) {
            for (int j = 0; j < DTW_DIM; ++j) {
                a[(size_t)i * DTW_DIM + j] = rng_gauss();
                b[(size_t)i * DTW_DIM + j] = rng_gauss();
            }
        }

        for (int band = 4; band <= DTW_MAX_BAND; band *= 2) {
            dtw_ctx_t c = {
                .a = a,
                .b = b,
                .row0 = row0,
                .row1 = row1,
                .len = DTW_LEN,
                .band = band,
                .slots = 2 * band + 3,
                .dist = 0.0f
            };

            bench_result_t r = run_benchmark(dtw_run, &c);

            /* Duas linhas de 2r+3 floats, e a memoria viva do DTW. */
            const size_t ram = (size_t)c.slots * 2u * sizeof(float);

            printf("DTW_RESULT,band=%d,slots_per_row=%d,us_per_comparison=%.3f,"
                   "cycles_per_comparison=%lu,ram_bytes=%lu,ram_kib=%.3f,"
                   "distance=%.4f\n",
                   band, c.slots, (double)((float)r.us),
                   (unsigned long)r.cycles, (unsigned long)ram,
                   (double)(ram / 1024.0), (double)c.dist);

            print_result("DTW", band, r, c.dist);
        }

        free(a);
        free(b);
        free(row0);
        free(row1);
    }

    ESP_LOGI(TAG, "Benchmark finished. sink_f=%f sink_i=%ld",
             (double)g_sink_f, (long)g_sink_i);
    ESP_LOGI(TAG, "Min free heap at end: %lu",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_8BIT));
}
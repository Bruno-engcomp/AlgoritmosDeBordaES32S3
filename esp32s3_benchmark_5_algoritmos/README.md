# Benchmark ESP32-S3 — 5 primeiros algoritmos

Este projeto mede cinco blocos computacionais muito comuns em aplicações embarcadas, processamento de sinais e redes neurais no ESP32-S3. A ideia é comparar custo computacional em ciclos, tempo e memória, usando a mesma plataforma para avaliar a eficiência relativa de cada operação.

## 1) FFT radix-2/4 (float32)

A FFT (Fast Fourier Transform) é a implementação rápida da Transformada de Fourier Discreta (DFT). Ela converte um sinal do domínio do tempo para o domínio da frequência.

Por que ela importa:
- permite identificar componentes frequenciais de um sinal;
- é usada em análise espectral, filtragem, detecção de ruído e processamento de áudio;
- sua complexidade é O(N log N), muito menor do que a DFT direta, que seria O(N²).

No benchmark:
- o tamanho do vetor varia de 64 a 4096 pontos;
- a métrica principal é ciclos por amostra ou ciclos totais em função de N;
- a FFT é baseada em `esp-dsp`.

Em termos simples: a FFT diz “quais frequências estão presentes no sinal”, e o benchmark mede o custo para fazer essa conversão na placa.

## 2) FIR (float32)

O filtro FIR (Finite Impulse Response) é um filtro digital que aplica uma soma ponderada de amostras recentes do sinal. Ele usa apenas entradas passadas e atuais, sem realimentação.

Fórmula básica:

```text
y[n] = a0*x[n] + a1*x[n-1] + a2*x[n-2] + ... + ak*x[n-k]
```

Por que ela importa:
- filtros FIR são estáveis e fáceis de projetar;
- têm resposta linear em fase;
- são usados em equalização, suavização, detecção e processamento de áudio.

No benchmark:
- o parâmetro é o número de taps (8, 16, 32, 64, 128, 256);
- o valor mais útil para relatório é ciclos/amostra;
- o filtro é implementado com suporte da biblioteca `esp-dsp`.

Em termos simples: o FIR “pesa” o sinal com coeficientes fixos e produz uma versão filtrada do sinal.

## 3) CNN 1D/2D (int8)

Uma CNN (Convolutional Neural Network) aplica filtros chamados kernels sobre a entrada para extrair características locais. Em vez de conectar todos os neurônios entre si, ela usa janelas menores e compartilhamento de pesos.

Por que ela importa:
- é muito usada em visão computacional, áudio, séries temporais e processamento de sinais;
- consegue detectar bordas, padrões e estruturas locais;
- reduz o número de parâmetros em comparação com redes totalmente conectadas.

No benchmark:
- a implementação de referência usa int8 em C, sem depender de TFLite Micro;
- a ideia é medir o custo de uma convolução neuronal básica;
- a métrica principal é tempo por inferência (ms/inferência) e FPS.

Importante:
- esta não é ainda uma comparação com TFLite Micro + ESP-NN;
- este é um baseline autocontido para estabelecer uma linha de referência.

Em termos simples: a CNN percorre a entrada com um filtro pequeno e gera um mapa de características.

## 4) IIR biquad

O filtro IIR (Infinite Impulse Response) é um filtro recursivo, ou seja, ele usa também saídas anteriores. O filtro biquad é um bloco de segunda ordem, extremamente comum em DSP.

Estrutura típica:

```text
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

Por que ela importa:
- permite projetar filtros com resposta seletiva precisa em poucos estágios;
- é muito usado em equalização, crossover de áudio e filtros analógicos equivalentes;
- oferece melhor eficiência do que FIR para filtros de alta seletividade com poucos coeficientes.

No benchmark:
- o número de seções varia de 2 a 8 biquads;
- o filtro é medido em ciclos/amostra para diferentes níveis de cascata;
- a implementação usa `esp-dsp`.

Em termos simples: um IIR combina entradas e saídas passadas para produzir um filtro mais eficiente, mas também mais sensível ao projeto.

## 5) MLP denso (int8)

A MLP (Multilayer Perceptron) densa, ou rede totalmente conectada, conecta cada entrada a cada neurônio da camada oculta e depois aos neurônios de saída.

Por que ela importa:
- é a base de redes neurais clássicas;
- é usada em classificação, regressão e processamento de vetores;
- é útil como referência para comparar custo computacional de inferência em inferência embarcada.

No benchmark:
- a implementação usa int8 em C e um modelo de referência simples;
- a métrica principal é tempo por inferência e uso de memória em ativações;
- mede a etapa de cálculo de camada densa e da saída final.

Em termos simples: cada neurônio soma produtos entre entrada e pesos, aplica uma função de ativação e gera um valor de saída.

## O que o projeto mede

O código produz saídas em CSV para comparar desempenho em diferentes configurações.

```text
CSV,FFT,64,1234,5.14,0.123
CSV,FIR,8,987,4.11,0.456
CSV,CNN,1,4321,18.00,0.789
CSV,IIR,2,3210,13.37,0.222
CSV,MLP,32,5432,22.63,0.333
```

Formato:

```text
CSV,algoritmo,parametro,ciclos,tempo_us,checksum
```

O checksum existe para impedir que o compilador elimine o cálculo por ser aparentemente inútil.

## Métricas sugeridas para relatório

Para cada algoritmo, gere uma tabela e um gráfico:

- FFT:
  - eixo X: N (número de pontos)
  - eixo Y: ciclos ou µs

- FIR:
  - eixo X: número de taps
  - eixo Y: ciclos/amostra

- CNN:
  - eixo X: tamanho do modelo/configuração
  - eixo Y: ms/inferência ou FPS

- IIR:
  - eixo X: número de seções
  - eixo Y: ciclos/amostra

- MLP:
  - eixo X: tamanho da camada
  - eixo Y: ms/inferência

## Importante sobre CNN e MLP

Nesta primeira versão, CNN e MLP usam implementações int8 em C para manter o projeto autocontido, fácil de compilar e reproduzir.

Isso NÃO é ainda uma comparação com TFLite Micro + ESP-NN. O objetivo desta versão é estabelecer um baseline antes de integrar um modelo real. O ESP-NN é uma biblioteca otimizada da Espressif para ESP32-S3, mas integrar um modelo TFLM real exige definir arquitetura do modelo, quantização e tensor arena.

FFT, FIR e IIR usam `esp-dsp`.

## Ambiente

Recomendado: ESP-IDF 5.5 ou 6.x e uma placa ESP32-S3.

O projeto foi feito para `esp32s3`.

## Compilar e gravar

No terminal do ESP-IDF:

```bash
cd esp32s3_benchmark_5_algoritmos

idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Troque `/dev/ttyACM0` pela porta da sua placa.

## Repetições

Por padrão são feitas 30 execuções de cada configuração. As 5 primeiras execuções são descartadas como warm-up.

Altere:

```c
#define BENCH_REPS 30
#define WARMUP_REPS 5
```

## CSV

No Linux você pode salvar o monitor:

```bash
idf.py monitor | tee resultado.txt
```

Depois extraia as linhas CSV:

```bash
grep '^CSV,' resultado.txt > resultados.csv
```

O arquivo poderá ser aberto no LibreOffice Calc/Excel.

## Boas práticas para benchmark

- use a mesma frequência de CPU em todas as medições;
- mantenha Wi-Fi/Bluetooth desligados;
- não imprima dentro do trecho cronometrado;
- faça warm-up;
- repita várias vezes;
- compare média/mediana, mínimo e desvio;
- use o mesmo conjunto de entrada;
- registre versão do ESP-IDF, placa e frequência da CPU.

O documento original informa as complexidades e métricas sugeridas; os resultados reais devem ser obtidos na placa.

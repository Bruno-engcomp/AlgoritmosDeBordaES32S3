# Benchmark ESP32-S3 — 5 primeiros algoritmos

Projeto ESP-IDF para benchmark no ESP32-S3 baseado nos 5 primeiros itens do documento fornecido:

1. FFT radix-2/4 (float32)
2. FIR (float32)
3. CNN 1D/2D (int8)
4. IIR biquad
5. MLP denso (int8)

O documento sugere, respectivamente, medir:
- FFT: ciclos vs N, de 64 a 4096 pontos;
- FIR: ciclos/amostra vs número de taps, de 8 a 256;
- CNN: ms/inferência e FPS;
- IIR: ciclos/amostra vs 2–8 seções;
- MLP: ms/inferência e SRAM de ativações.

## Importante sobre CNN e MLP

Nesta primeira versão, CNN e MLP usam kernels int8 de referência em C para que o projeto seja autocontido e fácil de reproduzir.

Isso NÃO é ainda uma comparação TFLite Micro + ESP-NN. O objetivo desta versão é estabelecer um baseline. O ESP-NN é uma biblioteca otimizada da Espressif para ESP32-S3, mas integrar um modelo TFLM real exige definir modelo, quantização e tensor arena.

FFT/FIR/IIR usam esp-dsp.

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

## Saída

O monitor imprimirá linhas CSV como:

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

## Boas práticas

Para benchmark:

- use a mesma frequência de CPU em todas as medições;
- mantenha Wi-Fi/Bluetooth desligados;
- não imprima dentro do trecho cronometrado;
- faça warm-up;
- repita várias vezes;
- compare média/mediana, mínimo e desvio;
- use o mesmo conjunto de entrada;
- registre versão do ESP-IDF, placa e frequência da CPU.

## O que você deve colocar no relatório

Para cada algoritmo, gere uma tabela e um gráfico.

Exemplos:

FFT:
- eixo X: N
- eixo Y: ciclos ou µs

FIR:
- eixo X: número de taps
- eixo Y: ciclos/amostra

CNN:
- eixo X: tamanho do modelo/configuração
- eixo Y: ms/inferência ou FPS

IIR:
- eixo X: número de seções
- eixo Y: ciclos/amostra

MLP:
- eixo X: tamanho da camada
- eixo Y: ms/inferência

O documento original informa as complexidades e métricas sugeridas; os resultados reais devem ser obtidos na placa.

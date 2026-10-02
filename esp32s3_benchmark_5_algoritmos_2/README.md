# Benchmark ESP32-S3 — algoritmos 6 a 10

Continuação de `esp32s3_benchmark_5_algoritmos`, que cobre os itens 1 a 5 do
documento. Aqui estão os itens 6 a 10:

| # | Algoritmo | Categoria | Implementação | Complexidade |
|---|-----------|-----------|---------------|--------------|
| 6 | MFCC (Mel + DCT) | PDS | esp-dsp (FFT + janela + DCT) | O(N log N) |
| 7 | Filtro de Kalman | Filtro | C puro (matrizes pequenas) | O(n³) |
| 8 | Goertzel | PDS | C puro (recorrência) | O(N) |
| 9 | Filtro de mediana | Filtro | C puro (janela ordenada) | O(N·k) |
| 10 | Média móvel | Filtro | C puro (soma deslizante) | O(N) |

Este README tem duas partes: **o que cada algoritmo é e para que serve**, e
**como compilar, medir e gerar os gráficos**.

---

# Parte 1 — Os algoritmos

Os cinco são **etapas de um pipeline real**, não alternativas. Typical:

```
microfone
  → Goertzel em 50 Hz        remove o zumbido da rede
  → Média móvel             suaviza o que sobrou
  → Mediana                 remove picos isolados
  → MFCC                    transforma áudio em vetor
  → (rede neural classifica)
```

E o Kalman entra em **qualquer** ponto, para deixar uma leitura confiável com
incerteza conhecida.

---

## 6. MFCC — Mel Frequency Cepstral Coefficients

### O que é

MFCC é a forma padrão de transformar um trecho de áudio em um vetor de poucos
números que descreve seu **timbre**. É a porta de entrada de quase todo
reconhecimento de fala.

A ideia central: um espectro de áudio é uma curva com centenas de valores, e
modelos de classificação esperam um vetor pequeno e com poucas correlações. O
MFCC entrega 13 a 26 números por quadro de áudio.

### Por que existe

Três problemas com usar o espectro direto:

1. **O ouvido humano não é linear em frequência.** Ele discrimina bem novas
   frequências baixas e mal frequências altas. A escala **mel** corrige isso:
   duas faixas de largura idêntica em Hz soam muito diferentes em 100 Hz e em
   4 kHz, mas iguais na escala mel.

   ```
   mel(f)  = 2595 · log10(1 + f/700)
   f(mel)  = 700 · (10^(mel/2595) − 1)
   ```

   São a mesma expressão em duas direções (a segunda é a inversa da primeira).

2. **A forma do espectro varia muito com o volume.** Um trecho de fala falado
   alto e o mesmo sussurrado têm a mesma forma, escala diferente. O logaritmo
   remove essa dependência de ganho.

3. **As informações relevantes são mal concentradas.** Os picos do espectro
   (os formantes, que dizem qual vogal é) estão espalhados ao longo de toda a
   curva. A DCT comprime essa informação e concentra a variação importante nos
   primeiros coeficientes.

### O pipeline, passo a passo

```
áudio
  ↓  1. janelamento        recorta quadros de 25–30 ms (aqui 30 ms = 480 amostras a 16 kHz)
  ↓  2. janela            Hann, suaviza as bordas do recorte
  ↓  3. FFT                domínio do tempo → domínio da frequência
  ↓  4. potência           |X[k]|² — a fase não carrega informação de envelope
  ↓  5. banco mel          26 filtros triangulares somando as frequências de interesse
  ↓  6. log                logaritmo de cada banda mel
  ↓  7. DCT                26 valores → 26 coeficientes = o MFCC
```

Os dois últimos passos juntos formam o **cepstrum**, que é uma "transformada
inversa": em vez de espectro virar sinal, ele vira espectro em outro espectro,
onde o pitch (altura da voz) aparece como um pico separado do timbre. É por
isso que se aplica *liftering*: cortar os primeiros coeficientes (que são os de
baixa queima, dominados pelo pitch) deixa só o timbre.

**O banco de filtros mel** é o heart do método. Não se calcula um valor por
frequência: calculam-se 26 valores, um por banda, e cada banda é a soma de
várias frequências com pesos triangulares. As bandas são igualmente espaçadas
em mel e se sobrepõem, o que dá transição suave em vez de degraus.

**Por que triangular?** É a forma mais simples que faz duas bandas vizinhas
darem exatamente a mesma contribuição na fronteira entre elas. Se as bordas
não casassem, o valor de cada banda dependeria de como os degraus foram
desenhados, e o resultado mudaria com pequenos ajustes de frequência.

### Onde é usado

- **Keyword spotting** — reconhecer "jarvis", "ok google" sem mandar áudio para
  a nuvem. É o caso clássico de borda, e o motivo de o MFCC entrar na lista:
  é o primeiro passo de qualquer interface de voz offline.
- **Verificação de locutor** — quem está falando.
- **Detecção de atividade de voz (VAD)** — distinguir silêncio de fala.
- **Classificação de som ambiente** — vidro quebrando, sirene, batida na porta.
- **Bioacústica** — classificar espécies a partir de cantos de pássaros.

### No projeto

`main.c:mfcc_run`. Configuração: 16 kHz, janela de 30 ms (480 amostras), 16
coeficientes mel, banco triangular de 80 Hz a 7600 Hz, DCT de 32 pontos
(precisa ser potência de 2 e maior que o número de coeficientes mel).

Medida: µs por janela de 30 ms.

### Notas de implementação

- O banco de filtros mel é montado **uma vez por tamanho de FFT, fora da região
  cronometrada** — só o pipeline é medido.
- O DCT do esp-dsp é o tipo II **sem escala** e exige um buffer de `N·2` floats,
  não `N`.
- Os coeficientes mel usam `log10` e somam um piso de `1e-10` para o log não
  estourar em quadros silenciosos.
- O IDF traz `dsps_wind_hann_f32`; a janela usada é Hann. O clássico de MFCC é
  Hamming, mas a escolha da janela altera a ordem de grandeza medida, não a
  comparação entre os algoritmos.
- Este baseline **não** aplica pré-ênfase, passo opcional e barato (um filtro
  de primeira ordem que compensa o rolloff do microfone). Não muda a ordem de
  grandeza, e omitir mantém a comparação limpa.
- O script imprime um `real_time_factor`: quantas janelas de 30 ms a placa
  processa por segundo. O que importa em voz não é só o tempo, é se dá para
  processar **mais rápido que o tempo real**, ou o sistema atrasa.

---

## 7. Filtro de Kalman

### O que é

Um estimador recursivo que mantém uma **estimativa de estado** e, mais
importante, **quão incerta essa estimativa é**. Alterna dois passos, para
sempre:

1. **Predizer** — usando o modelo do sistema, para onde o estado deveria estar
   agora, e como a incerteza cresceu no caminho.
2. **Atualizar** — comparando a previsão com a medição real, e corrigindo
   proporcionalmente a quem se pode confiar mais.

A grande vantagem sobre um filtro comum: **ele decide o quanto corrigir**.
Quando a medição é ruidosa ele quase a ignora; quando é confiável ele segue
ela. Nenhum ajuste manual.

### A matemática

Com estado `x`, medição `z` e matriz de medição `H`:

**Predição**
```
x⁻ = F · x            P⁻ = F · P · Fᵀ + Q
```

**Atualização**
```
S   = H · P⁻ · Hᵀ + R
K   = P⁻ · Hᵀ · S⁻¹
x   = x⁻ + K · (z − H · x⁻)
P   = (I − K · H) · P⁻
```

| Símbolo | Nome | Significado |
|---------|------|-------------|
| `F` | transição de estado | como o estado evolui sem medição |
| `H` | observação | como o estado vira uma medição |
| `Q` | ruído do processo | incerteza do modelo |
| `R` | ruído da medição | incerteza do sensor |
| `P` | covariância | incerteza atual, e o Kalman a **propaga** |
| `K` | ganho de Kalman | quanto confiar na medição neste instante |

`K` é a peça central: `K = P⁻Hᵀ/(H P⁻Hᵀ + R)`. Se `R` (sensor ruim) domina,
`K → 0` e o filtro quase não corrige. Se `P⁻` (modelo confiável) domina,
`K → 1` e o filtro basicamente segue a medição. O Kalman **não precisa de você
escolher** esse equilíbrio.

O custo é `O(n³)`, das multiplicações matriz por matriz.

### Onde é usado

- **Fusão de IMU** — junta acelerômetro e giroscópio, que se complementam: o
  giroscópio tem boa resposta em alta frequência mas integra offset; o
  acelerômetro não tem drift mas é barulhento. Esta é a aplicação mais comum
  em borda, em drones e wearables.
- **GPS + IMU** — entrega solução contínua mesmo quando o GPS perde sinal, e
  sabe dizer o quanto confiar nela.
- **Suavização com incerteza** — temperatura, tensão de bateria, nível de
  tanque, onde a média móvel não sabe se a leitura atual é confiável.
- **Estimação de estado** — posição, velocidade, inclinação.

### Limitações e variantes

O Kalman clássico **exige linearidade** e ruído gaussiano. Quando o modelo é
não linear (medição de distância, sensor angular), usa-se o **EKF**, que
lineariza em torno da estimativa atual com uma Jacobiana. O custo sobe e a
Jacobiana precisa ser derivada à mão, que é onde a maioria dos erros aparece.

A propósito: o esp-dsp **tem** um EKF pronto, mas em **C++** (`ekf.h`, classe
`ekf` com `dspm::Mat`). O documento pede Kalman em C puro, então aqui é em C,
para ficar autocontida e comparável com as demais.

### No projeto

O sistema de `n` estados é uma cadeia: cada estado recebe o estado seguinte, e
só o estado medido (o último) persiste. É um sistema linear estável, todos os
estados convergem para a medição, e a covariância fica bem condicionada em
float32 até `n = 8`.

Medida: ciclos por atualização, varrendo `n` de 2 a 8.

### Notas de implementação

Dois erros que **precisam** ser evitados, e que não aparecem como crash — eles
só se manifestam muito depois:

- **A atualização de `P` tem que ser fora do lugar.** Ler `P[i][n-1]` e
  `P[n-1][j]` enquanto `P` está sendo reescrita corrompe o ganho de Kalman e
  faz a covariância perder a simetria. Os estados divergem silenciosamente.
- **Só o estado medido pode persistir.** Dar persistência a todos os estados,
  como num modelo de velocidade constante, faz cada estado atrasar o de cima
  por um fator `1/(1−α)`, e a estimativa cresce exponencialmente com `n`
  (cerca de 10⁶ em `n = 8`). Isso estoura a precisão do float32.

---

## 8. Algoritmo de Goertzel

### O que é

Uma forma de computar a transformação de Fourier de um sinal, mas **produzindo
somente um binário de frequência**. É o oposto da FFT: a FFT calcula todas as
frequências de uma vez, o Goertzel calcula uma frequência específica com o
custo de uma varredura simples.

### A recorrência

Em vez do somatório da DFT, usa-se uma **recorrência de segunda ordem** com
dois acumuladores `s1`, `s2`:

```
s[n] = x[n] + 2·cos(ω)·s[n−1] − s[n−2]
```

O coeficiente `2·cos(ω)` é **constante** (depende só da frequência alvo e da
taxa de amostragem), então o laço interno é um par de multiplicação e subtração
por amostra. Ao final, a potência do bin alvo sai da fórmula fechada:

```
P = s[N−1]² + s[N−2]² − 2·cos(ω)·s[N−1]·s[N−2]
```

### Por que usar em vez da FFT

| | FFT | Goertzel |
|---|---|---|
| Custo | `O(N·log N)` | `O(N)` |
| Memória | `O(N)` (precisa manter o buffer) | `O(1)` (dois acumuladores) |
| Saída | todas as frequências | uma frequência |

Se você quer **um** tom, a FFT faz todo o trabalho extra de graça. É a
otimização clássica: não calcule o que você não vai usar.

O mesmo raciocínio do MFCC em miniatura — aqui o número de frequências de
interesse é 1, lá são 26 bandas.

### Onde é usado

- **DTMF** — os tons dos botões do telefone. São 8 frequências (2 grupos de 4),
  então dá para rodar 8 Goertzels em paralelo, ou uma FFT pequena. É o exemplo
  mais famoso.
- **Detecção de pitch** — a frequência fundamental de uma nota musical.
- **Detecção de wake word** — medir energia numa faixa e disparar.
- **Ruído de rede elétrica** — 50 Hz (Brasil) e 60 Hz. Um Goertzel em 50 Hz é
  muito mais barato que filtrar a faixa toda, e é o caso mais direto em borda:
  o pico de 50 Hz contamina qualquer leitura de sensor.
- **Sensores ultrassônicos / infravermelho** — muitos protocolos de pulso
  detectam a resposta medindo energia na frequência de emissão.

### No projeto

`main.c:goertzel_run`. O sinal de entrada é um tom puro de 1000 Hz a 16 kHz, e
o teste de detecção é 50× mais sensível na frequência alvo do que numa
frequência errada.

Medida: ciclos por amostra, varrendo o tamanho do bloco de 256 a 4096.

---

## 9. Filtro de mediana

### O que é

Filtro **não linear**: em vez de substituir cada amostra pela média da janela,
substitui pela **mediana** (o valor central depois de ordenar a janela).

### Por que não usar a média

A média é vulnerável a outliers. Numa janela de 5 amostras
`[10, 11, 12, 13, 513]`, a média é 111,8 — um único valor corrompido arruína
toda a janela, e o erro se espalha por ela. A mediana é 12: o pico é
**descartado**.

Esse padrão de dado contaminado tem nome próprio: **ruído impulsivo**, ou
*salt and pepper* ("sal e pimenta") — valores isolados e errados cercados de
valores bons. É bem diferente de ruído gaussiano, que se espalha por todos os
pontos.

### O que ele ganha e o que ele perde

**Ganha:**

- Remove picos isolados de forma seletiva, ao contrário da média.
- **Preserva bordas.** Numa transição rápida de valor, a média arredonda a
  transição; a mediana mantém os dois lados nítidos.
- Não inventa valores novos: a saída é sempre uma das amostras de entrada.
- Mais resistente que a média a distribuições não normais.

**Perde:**

- Custa mais que a média (ordenar não é uma subtração e uma adição).
- Rejeita bem menos ruído gaussiano, que é o ruído mais comum.
- Em **rampas** ele suaviza, porque a mediana de uma rampa é o valor central
  dela, não o valor da posição atual. Ele "arrasta" a borda para trás.
- Tem **memória mais longa**, então adiciona mais atraso no sinal.

### Onde é usado

- **Sensores com picos** — acelerômetro com solavanco, sensor ultrassônico com
  eco perdido. É muito comum o valor vir absurdo num instante, e a mediana o
  remove sozinho.
- **Remoção de "clique" em áudio** — o artefato característico do truncamento
  de um arquivo de áudio é um estalo, que é justamente um pico.
- **Despiking em imagens** — remove pixels de sal e pimenta preservando as
  bordas (ao contrário da média, que só borra).
- **Pré-processamento de visão computacional** — antes de limiarização,
  segmentação ou detecção de bordas, onde um pixel outlier vira um falso objeto
  inteiro.
- **Cálculo de altura de onda e nível de líquido** — leitura por ultrassom com
  eco ocasionalmente perdido.

### No projeto

`main.c:median_run`. Entrada: rampa suave de 4800 amostras com picos de +500 a
cada 64 amostras. A saída bate com a mediana de força bruta da mesma janela
para todo `k` de 3 a 15.

Medida: ciclos por amostra, varrendo `k` de 3 a 15.

### Notas de implementação

**Não dá para manter a janela ordenada e continuar.** Ordenar a janela destrói
a ordem de chegada das amostras, e a amostra mais antiga não é necessariamente
a menor nem a maior da janela — então não há como saber qual posição
descartar. A implementação ingênua mantém o primeiro elemento congelado e a
saída trava num valor fixo (foi o primeiro bug real aqui, e ele não dá
crash: só dá números errados).

A solução é um **buffer circular** que guarda os últimos `k` valores na ordem
de chegada, mais uma cópia ordenada recalculada a cada amostra por inserção.
Como a janela já está quase ordenada (só um elemento muda por passo), a
inserção custa `O(k)` de verdade — dando `O(N·k)` amortizado, como o
documento sugere.

---

## 10. Média móvel

### O que é

O filtro mais simples que existe:

```
y[n] = (1/M) · (x[n] + x[n−1] + x[n−2] + … + x[n−M+1])
```

Cada saída é a média das últimas `M` amostras. É um FIR passa-baixa de resposta
ao impulso retangular, um **boxcar**.

### O truque que faz ser barata

A conta ingênua é `O(M)` por amostra, ou seja, `O(N·M)` no total, e o custo
cresce com o tamanho da janela. Mas a soma de uma janela deslizante tem muita
redundância: ao passar para a próxima amostra, **um valor sai e um entra**. A
soma é a mesma com uma subtração e uma adição:

```
soma[n] = soma[n−1] − x[n−M] + x[n]
```

Com isso, a média móvel custa **`O(1)` por amostra** — custo constante,
independente do tamanho da janela. `O(N)` no total, o melhor possível.

É o mesmo raciocínio de "não calcule o que você não vai usar" do Goertzel, em
outra forma: aqui se reaproveita o que já foi calculado.

### Propriedades

- Resposta em frequência: `sinc`, com nulos nas múltiplas de `fs/M`.
- **Atraso de grupo** de `(M−1)/2` amostras. É causal, então há atraso
  inerente: aumentar a suavização sempre custa latência. É a troca
  fundamental de qualquer média móvel.
- Para uma rampa perfeitamente linear, a média móvel passa exatamente pela
  rampa, mas atrasada em `(M−1)/2`. É por isso que serve para remover ruído
  sem distorcer a tendência.
- Em alta frequência, atenua proporcionalmente à frequência.

### Onde é usado

- **Suavização genérica** de qualquer leitura de sensor.
- **Baseline de custo.** É o motivo de ele estar na lista: é o **custo mínimo
  de referência** de um filtro em ESP32-S3. Se um filtro fino custa 300
  ciclos/amostra e a média móvel custa 8, sabe-se o preço da complexidade.
- **Pré-estimador** — suavizar antes de derivar, para amplificar um sinal que
  já foi limpo.
- **Monitoramento de bateria e temperatura**, onde a latência de meia janela é
  irrelevante.
- **Taxa de amostragem** — por média, entender a média.

### Quando não usar

Se há outliers, a **mediana** (item 9) é melhor. Se o filtro precisa preservar
bordas, nenhum dos dois é adequado. Se a latência importa, a média móvel causa
atraso. Para resposta em fase com o sinal, a versão **não causal** com
centralização da janela elimina o atraso, ao custo de olhar para o futuro
(só é válido com o lote inteiro disponível).

### No projeto

`main.c:ma_run`. A saída confere com a média recalculada do zero na mesma
janela, para todo tamanho de 4 a 256.

Medida: ciclos por amostra, varrendo a janela de 4 a 256.

### Relação com a mediana

Média e mediana estão na mesma lista por motivos diferentes. A média é barata e
vulnerável a outliers. A mediana é robusta e mais cara. **Mediana × média é a
escolha entre robustez e custo**; nenhuma é correta no geral.

---

## Comparando os cinco

| Algoritmo | Custo | Tipo | Robusto a outliers | Memória |
|-----------|-------|------|--------------------|---------|
| MFCC | alto | PDS | não é filtro | `O(N)` |
| Kalman | alto, `O(n³)` | recursivo | pela covariância | `O(n²)` |
| Goertzel | baixo, `O(N)` | PDS | não é filtro | `O(1)` |
| Mediana | médio, `O(N·k)` | não linear | **sim** | `O(k)` |
| Média móvel | mínimo, `O(N)` | linear | não | `O(M)` |

## Por que medir no ESP32-S3

O custo em ciclos no desktop é quase irrelevante para o projeto de borda. No
ESP32-S3 o que importa é:

- **O custo é o orçamento de tempo.** Para áudio a 16 kHz, uma amostra a cada
  62,5 µs. Se o filtro passar disso, o sistema atrasa.
- **RAM é o orçamento de memória.** Buffers grandes consomem SRAM, que é
  escasso. O Goertzel usar `O(1)` em vez de `O(N)` pode ser a diferença entre
  caber e não caber.
- **SIMD.** O esp-dsp usa as extensões SIMD da CPU; uma versão ingênua em C
  puro pode ser várias vezes mais lenta que a otimizada. FFT, FIR e IIR são o
  caso mais claro.
- **Determinismo.** Na borda, a latência previsível importa tanto quanto a
  média.

---

# Parte 2 — Como usar o projeto

## Ambiente

ESP-IDF 5.5 ou 6.x e uma placa ESP32-S3. Compilado e validado com
**ESP-IDF v6.0.2** para o alvo `esp32s3`.

## Compilar e gravar

```bash
cd esp32s3_benchmark_5_algoritmos_2

idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Se `idf.py` não estiver no PATH, carregue o ambiente antes:

```bash
source ~/.espressif/v6.0.2/esp-idf/export.sh
```

Troque `/dev/ttyACM0` pela porta da sua placa.

## Saída

O monitor imprime linhas CSV:

```text
CSV,MFCC,256,45123,152,1.234567
CSV,KALMAN,2,3120,10,0.998765
CSV,GOERTZEL,256,14,0,0.765432
CSV,MEDIAN,3,88,0,110.123456
CSV,MAVG,4,9,0,0.654321
```

Formato:

```text
CSV,algoritmo,parametro,ciclos,tempo_ns,checksum
```

Para Goertzel, mediana e média móvel o tempo já vem normalizado por amostra
(em ns). Para MFCC é por janela de 30 ms. Para Kalman é por atualização.

O checksum existe para impedir que o compilador elimine o cálculo por ser
aparentemente inútil.

Além do CSV, cada algoritmo imprime uma linha de resumo com uma métrica
secundária, útil para conferir o relatório:

```text
MFCC_RESULT,fft_size=512,ms_per_30ms_window=0.152,cycles=45123,real_time_factor=197.36
KALMAN_RESULT,states=4,cycles_per_update=8450,us_per_update=35.208,state_estimate=1.000000
```

## Repetições

Por padrão são feitas 30 execuções de cada configuração. As 5 primeiras são
descartadas como warm-up.

Altere em `main/main.c`:

```c
#define BENCH_REPS 30
#define WARMUP_REPS 5
```

## CSV e gráficos

No Linux, salve o monitor:

```bash
idf.py monitor | tee resultado.txt
```

Depois extraia as linhas CSV:

```bash
grep '^CSV,' resultado.txt > resultados.csv
```

Para gerar os gráficos e a tabela resumida:

```bash
python3 -m pip install matplotlib
python3 analyze_results.py
```

O script usa o backend `Agg`, então roda sem display e salva os PNGs
(`mfcc_ciclos.png`, `kalman_ciclos.png`, etc.) mais `resultados_resumido.csv`.
A escala X dos filtros por amostra é log na base 2, porque a janela varia por
potências de 2.

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

O `sdkconfig.defaults` já fixa CPU em 240 MHz, otimização de performance e
flash de 4 MB.

## O que você deve colocar no relatório

Para cada algoritmo, gere uma tabela e um gráfico:

- **MFCC**: eixo X tamanho da FFT, eixo Y µs por janela de 30 ms;
- **Kalman**: eixo X nº de estados, eixo Y ciclos por atualização;
- **Goertzel**: eixo X tamanho do bloco, eixo Y ciclos por amostra;
- **Mediana**: eixo X janela k, eixo Y ciclos por amostra;
- **Média móvel**: eixo X janela, eixo Y ciclos por amostra.

O documento original informa as complexidades e métricas sugeridas; os
resultados reais devem ser obtidos na placa.

## Referências

- Lyons, *Understanding Digital Signal Processing*
- Proakis & Manolakis, *Digital Signal Processing: Principles, Algorithms and
  Applications*
- Haykin & Moher, *Communication Systems*
- University of Adelaide, *Kalman filter* — notas de aula
- Espressif, *ESP-DSP Programming Guide*
  (`managed_components/espressif__esp-dsp`)

## Continuação

Os itens 11 a 16 do documento (LMS, SVM, Random Forest, k-NN, DWT, DTW) ainda
não foram implementados.

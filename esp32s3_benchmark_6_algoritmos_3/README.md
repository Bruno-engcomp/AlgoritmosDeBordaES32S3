# Benchmark ESP32-S3 — algoritmos 11 a 16

Terceira e última parte da série. `esp32s3_benchmark_5_algoritmos` cobre os
itens 1 a 5 e `esp32s3_benchmark_5_algoritmos_2` cobre os itens 6 a 10. Aqui
estão os seis últimos:

| # | Algoritmo | Categoria | Implementação | Complexidade |
|---|-----------|-----------|---------------|--------------|
| 11 | LMS adaptativo | Filtro | C puro (NLMS) | O(N·M) |
| 12 | SVM (linear/RBF) | Classificação | C puro (laço sobre SV) | O(D·SV) |
| 13 | Random Forest | Classificação | C puro (árvore if-else) | O(T·P) |
| 14 | k-NN | Classificação | C puro (distância euclidiana) | O(N·D) |
| 15 | DWT (Haar / db4) | PDS | C puro (banco de filtros) | O(N) |
| 16 | DTW | Classificação | C puro (programação dinâmica) | O(N·M) |

Este README tem duas partes: **o que cada algoritmo é e para que serve**, e
**como compilar, medir e gerar os gráficos**.

Com estes seis, os 16 itens do documento estão implementados.

---

# Parte 1 — Os algoritmos

Os seis não formam um pipeline como os cinco do projeto anterior. São
escolhas **alternativas** para o mesmo problema: **receber um vetor de
características e dizer a que classe ele pertence**. Três deles (SVM, Random
Forest, k-NN) classificam com aprendizado de máquina.

```
vetor de características  (ex.: 13 coeficientes MFCC)
   → SVM          borda reta / decisão por Kernel
   → RandomForest muitas árvores, cada uma vota
   → k-NN         vizinhos mais próximos
```

Os outros três são pré-processamento e medição:

- **LMS** filtra ruído **que ainda não se sabe** de antemão;
- **DWT** decompõe o sinal em faixas (é o pré-processamento natural de quem
  quer classificadores robustos a deslocamentos no tempo);
- **DTW** alinha duas séries temporais, e é a forma padrão de comparar gestos
  de aceleração.

O tema de todos é o mesmo que domina a série: **quanto custa na placa** e
**cabe na SRAM**.

---

## 11. LMS adaptativo

### O que é

Um filtro FIR comum tem coeficientes **fixos**, resolvidos uma vez na
fabricação. O LMS é um filtro cujos coeficientes são **reajustados a cada
amostra**, a partir do erro que ele mesmo comete.

Cada amostra, o filtro faz três coisas:

1. **Filtra** — `y[n] = Σ w[m]·x[n−m]`
2. **Compara** com a referência desejada `d[n]` e mede o erro `e[n] = d[n] − y[n]`
3. **Corrige** os coeficientes na direção que reduz esse erro

```
e[n] = d[n] − y[n]
w[m] += μ · e[n] · x[n−m]      (para todo m)
```

É o algoritmo de **Widrow-Hoff** (1960), e é a ideia que está por trás de
cancelamento de eco, cancelamento de ruído, equalização de canal e antenas
adaptativas. Todos são o mesmo mecanismo.

### Por que NLMS e não LMS

O passo `μ` é o único parâmetro, e a estabilidade depende dele:

```
0 < μ < 2 / (M · σx²)
```

Com sinal forte (`σx²` grande) ou muitos coeficientes (`M` grande), o limite
superior cai e **o filtro diverge em float32**. Em vez de escolher `μ`
certinho para cada sinal, o **NLMS** divide pela potência instantânea da
janela de entrada:

```
w[m] += (μ / (Σ x² + ε)) · e[n] · x[n−m]
```

Com isso o passo efetivo fica sempre na faixa `(0, 2)`, **independente de
`M` e da potência do sinal**. É por isso que aqui `μ = 0.5` é constante em
toda a varredura — e é o motivo de o código medir o NLMS e o documento dizer
LMS.

O `ε` no denominador existe porque, nas primeiras amostras, a janela ainda
está quase zerada e a divisão estouraria.

### Onde é usado

- **Cancelamento de eco** — o microfone ouve o próprio alto-falante. A referência
  é o que foi enviado; o filtro aprende a subtrair o eco. É a aplicação
  mais comum em mãos-livres, VoIP e videochamada.
- **Cancelamento de ruído** — microfone de proximidade ou anti-ventilamento.
- **Equalização de canal** — a resposta do canal muda com o ambiente, e o
  filtro aprende acompensation sem intervenção.
- **Antenas adaptativas** — o LMS escolhe os pesos que maximizam o sinal à
  frente do ruído.
- **Ruido de saída em atlas** — o item 8 (Goertzel em 50 Hz) é um filtro
  **fixo** para uma interferência conhecida; o LMS generaliza o caso em que a
  interferência varia.

### Onde o projeto mede

`main.c:nlms_run`. **Identificação de sistema**: há um sistema desconhecido de
8 taps, e a referência `d` é a saída desse sistema para o mesmo ruído. O LMS
nunca vê os coeficientes verdadeiros — só entrada e referência. Ruído branco
como excitação, 4096 amostras, `M` de 8 a 128 taps.

Medida: **ciclos por amostra** em função de `M`.

### Notas de implementação

- **A janela é acessada por ponteiro** (`xi = &x[t]`), não por buffer circular
  nem `memmove`. É a forma mais rápida, porque o laço interno vira uma
  varredura contígua e vetorizável. O filtro é **causal**: `xi[M−1]` é a
  amostra atual.
- **Os pesos são zerados a cada execução.** Se não fossem, o filtro já
  adaptado pararia de se mexer, o trabalho somaria quase nada e a medição
  seria de um filtro parado. O que se mede é o custo de adaptar `N` amostras
  desde o início.
- **São dois laços de `M` por amostra**, não um: um para a saída e outro para
  atualizar os pesos. Por isso o custo por amostra cresce praticamente
  dobrado em `M`, e o relatório mostra o valor absoluto.
- **A potência sai de graça** do primeiro laço. Calcular `Σ x²` seria um
  terceiro laço; como `y[n]` já passa pelos mesmos `x[n−m]`, a soma sai no
  mesmo custo.
- **Com excitação sem ruído de medição, o NLMS converge até a precisão da
  máquina** (`||w − h||² ≈ 10⁻¹⁵`, erro relativo `10⁻¹⁵`). O
  `residual_mse_db` impresso é por isso dominado pelo **transiente** de
  convergência, e não por ruído de gradiente em regime. A **taxa de
  convergência** pedida pelo documento aparece em `iters_to_20db`.
- `iters_to_20db` é medido numa passagem **não cronometrada**: o laço principal
  não pode carregar a gravação de um traçado.

---

## 12. SVM — Support Vector Machine

### O que é

Dado um conjunto de pontos com duas classes, o SVM procura a **separação com
maior margem** entre elas. O critério é geométrico, não estatístico: ele não
precisa saber como os dados foram gerados, apenas que as classes não se
misturam.

A consequência prática é que o modelo final é **minúsculo**: em vez de guardar
os exemplos de treino, guarda só os pontos que definem a fronteira — os
**vetores-suporte** — e os seus pesos.

```
f(x) = Σ_i  α_i · y_i · K(x, sv_i) + b
```

`sv_i` é o vetor-suporte, `α_i` o peso, `y_i` o rótulo (±1) e `K` o **kernel**.
A soma é sobre os vetores-suporte, e é por isso que o custo é `O(D·SV)`: o
número de vetores-suporte é o parâmetro que decide se o modelo cabe na MCU.

### Os dois kernels

**Linear**: `K(x, s) = x·s`. Separa por uma reta.

**RBF** (RBF = *Radial Basis Function*, função de base radial):
`K(x, s) = exp(−γ·||x − s||²)`. Separa por uma **superfície**, e na prática
resolve tudo que a reta não resolve — é o padrão para áudio.

`γ` controla o alcance: pequeno dá uma superfície suave e que generaliza
demais; grande dá uma que contorna cada ponto de treino e **não generaliza**.
O valor clássico é `γ = 1/(D·σ²)`, e aqui `γ = 1/D` com `σ² = 1`.

### A otimização que importa em borda

Para o kernel linear, a soma pode ser reescrita:

```
f(x) = (Σ_i α_i · y_i · sv_i) · x + b
        └─────── w ────────┘
```

Com `w` pré-computado, a inferência custa **`O(D)`** em vez de `O(D·SV)` — e,
mais importante, **a base de vetores-suporte deixa de precisar estar na SRAM**.
Em um classificador de áudio com 128 vetores de 13 dimensões, isso são 6,5 KB
que não precisam existir em RAM.

O projeto **mede as duas formas** e imprime a diferença:

```
SVM_RESULT,n_sv=128,us_per_inference=...,us_collapsed=...,
              dual_vs_collapsed_diff=...,accuracy=...
```

`dual_vs_collapsed_diff` tem de sair em ~`10⁻⁷` (erro de arredondamento
float32): é a prova de que a reescrita é a mesma função.

### Onde é usado

- **Keyword spotting** — classificar vetores MFCC (13 coeficientes) em
  "palavra detectada / silêncio". É o exemplo mais direto dos itens 6 a 16
  juntos: o MFCC do projeto 2 alimenta o classificador daqui.
- **Detecção de atividade de voz** — duas classes, é o problema mais simples.
- **Classificação de sensores** — aceleração, temperatura, vibração, com poucas
  amostras por classe.
- **Diagnóstico industrial** — a superfície de decisão é interpretável, o que
  conta quando o modelo precisa ser explicado.

### Quando não usar

Com muitas classes, a estratégia one-vs-rest multiplica o número de
inferências. E o kernel RBF **não** tem a otimização da forma reduzida: a
superfície depende de todos os vetores-suporte, e eles têm de ficar em RAM.

### Onde o projeto mede

`main.c:svm_decide`. `D = 13` (dimensão típica de MFCC), duas classes
separadas, varredura de 8 a 128 vetores-suporte, kernel linear e RBF.

Medida: **µs por inferência** em função do número de vetores-suporte.

### Notas de implementação

- **A distância vai ao quadrado, sem raiz.** A raiz é monótona e não muda a
  ordem, então o resultado é idêntico e economiza-se uma raiz por
  vetor-suporte.
- **No RBF o gargalo é o `expf`,** não a distância. Por isso o RBF custa
  cerca de 2x o linear, e não 10x, mesmo tendo o mesmo laço de `D`
  multiplicações.
- **O enxame de dados é separável por construção** (dois blocos em ±0.6). Isso
  mantém a acurácia alta e **isola o custo da inferência** do custo de um
  modelo ruim: com modelo ruim, medir só tempo não diz nada sobre o algoritmo.

---

## 13. Random Forest

### O que é

Um ensemble de árvores de decisão. Cada árvore é treinada em uma amostra
diferente dos mesmos dados (**bootstrap**, com reposição), e cada árvore é
limitada a um subconjunto aleatório de atributos (**mtry**) ao escolher onde
cortar. A resposta é a **votação** das árvores.

O ponto todo: uma árvore única é um classificador fraco e instável — muda
muito com uma pequena mudança nos dados. Mas árvores crescidas assim são
**pouco correlacionadas**, e a variância de uma média de modelos pouco
correlacionados cai muito. O ensemble é mais estável sem introduzir viés
grande.

### Por que vira `if-else`

Uma árvore treinada é uma sequência de comparações. Como a inferência só
precisa do caminho **para aquele vetor**, não da árvore inteira, dá para
achatar em código:

```c
if (x[3] <= 0.412f) node = 12;
else                node = 27;
```

É exatamente o que o emlearn gera, e é o que roda rápido na borda: sem
ponteiros, sem call, sem alocação. O custo é previsível, porque a
profundidade máxima limita o número de desvios.

### O que o algoritmo custa de verdade

O documento diz `O(T·P)`, e o detalhe importante é que **esse é o pior caso**.
Na inferência, cada nó testa **um** atributo contra um limiar — o atributo já
foi escolhido no treino — então uma árvore custa `profundidade` comparações,
não `P`:

```
inferência  =  T × profundidade        (≤ T × profundidade × D, se alguma árvore for profunda)
treino      =  T × nós × (D-1) × log(N)  ← aqui sim entra o (D-1): cada limiar
                                         é procurado testando todos os atributos
```

Numa floresta rasa (`profundidade = 5`), mesmo com 64 árvores, são ~320
comparações no total — barato. Já a **memória do modelo** é o que pesa: cada
nó guarda um índice de atributo, um limiar e dois índices de filho.

### Onde é usado

- **Sensores e série temporal tabular** — o uso clássico, porque a Random
  Forest **lida com atributos heterogêneos e não precisa de normalização**.
- **Manutenção preditiva** — a vibração de um motor vira features e a floresta
  classifica "normal / degrading".
- **Reconhecimento de gestos** — features de IMU combinadas com janela deslizante.
- **Qualquer lugar onde a acurácia importa mais que o tamanho do modelo** —
  ela quase nunca é a mais rápida, e isso é aceito de propósito.

### Quando não usar

Se **memória flash** é o aperto. É o modelo mais "gordo" da lista depois do
k-NN. E em tarefas multidimensionais com poucas amostras por classe, o
bagging com reposicional ajuda pouco e custa caro.

### Onde o projeto mede

`main.c:rf_run`. Treinamento de verdade: bootstrap de 256 amostras, busca do
melhor limiar por **Gini**, `mtry = 1 + floor(sqrt(13)) = 4`, profundidade
máxima 5, duas classes. Varredura de 8 a 64 árvores.

Medida: **µs por inferência** e **bytes do modelo** em função do número de
árvores.

### Notas de implementação

- **O treino é de verdade, aqui.** Bootstrap, busca de limiar por Gini e
  subconjunto de atributos — tudo montado antes da região medida. Isso
  importa: uma floresta "de mentira", com cortes arbitrários, mediria a
  estrutura de `if-else` e não o algoritmo.
- **As árvores são prefixo umas das outras.** O gerador determinístico é
  regerado com a mesma semente a cada montagem, então a floresta de 8 árvores
  é **exatamente** as 8 primeiras da de 64. Sem isso, comparar 8 contra 64
  seria comparar dois modelos diferentes.
- **Cada árvore ocupa uma fatia de 63 nós** (`2^(5+1) − 1`), para que os nós
  de uma árvore fiquem contíguos e a inferência ande por deslocamento fixo. É
  desperdício de RAM, mas deixa o custo de inferência previsível.
- **O custo do modelo é medido em bytes de nó, não em RAM alocada.** Com
  `tree_size[t]` Guardando só os nós usados, o número impresso é o que
  realmente ocuparia a flash num modelo gerado como `if-else`. A RAM
  alocada aqui é bem maior, porque é um array de nós de 20 bytes cada com
  folga para a profundidade máxima.
- **Os índices de amostras ficam em buffers estáticos por profundidade, não na
  pilha.** Alocar `2 × 256` inteiros por nível custaria 2 KB por frame, e a
  task principal do FreeRTOS tem pilha curta.

---

## 14. k-NN

### O que é

O classificador mais simples da lista, e o mais honesto: **não existe fase de
treino**. A base de treino *é* o modelo. Para classificar, calcula-se a
distância de `x` a cada ponto da base e toma-se o rótulo dos `k` mais
próximos, por maioria.

```
distância(x, s) = ‖x − s‖²
```

Nenhum parâmetro para ajustar, nenhuma hipótese sobre a distribuição. O único
compromisso é `k`: `k = 1` segue um vizinho só e é ruidoso demais; `k` grande
suaviza demais e apaga fronteiras.

### Por que é o mais caro dos seis

Dois motivos somados:

**Tempo:** cada inferência percorre **a base inteira**, `O(N·D)`. Com 1024
vetores de 13 dimensões, são 13 mil subtrações e multiplicações — por
inferência.

**Memória:** a base precisa estar **na SRAM**. Para 1024 vetores de 13
dimensões float32, 52 KB. Isso não é detalhe: é a base inteira ocupando a
memória mais escassa da placa.

E é por isso que o documento pede **ms por inferência contra o tamanho da
base**: o algoritmo não tem como ficar mais rápido sem mudar de watts
(método), mudar a representação (PCA, quantização) ou aceitar um índice
(árvore KD, ball tree) que deixa de ser k-NN puro.

### Onde é usado

- **Reconhecimento de gestos** — os prototypes do gesto viram a base. É
 classicamente o primeiro classificador que dá resultado em IMU.
- **Detecção de anomalia** — `k` vizinhos **distantes** é anomalia. Bastante
  aplicado em série temporal de sensores.
- **Classificação de assinatura / letra** — o stylus desenha, e o traço é
  comparado com os exemplos.
- **Fase de protótipo** — sem treino, sem infraestrutura. Serve para
  descobrir se o problema é separável antes de investir num modelo.

### Quando não usar

- **Muitas dimensões.** É a "maldição da dimensionalidade": em vetores de alta
  dimensão as distâncias se concentram, todas as distâncias ficam quase iguais
  e o vizinho mais próximo deixa de ser informativo.
- **Memória apertada** — a base é o modelo.
- **Inferência frequente** — o custo se paga toda vez.

### Onde o projeto mede

`main.c:knn_run`. `D = 13` (12 atributos + rótulo), base de 64 a 1024
vetores, `k = 3`, classes separadas por construção.

Medida: **µs por inferência** em função do tamanho da base.

### Notas de implementação

- **Distância ao quadrado, sem raiz.** A raiz é monótona e não muda a ordem,
  então o conjunto de vizinhos é o mesmo. Omite-se `N` raízes por inferência.
- **Descarte antecipado.** Se a distância do candidato já é pior que a do
  pior vizinho do top-`k`, o ponto é ignorado **antes** de escrever na lista.
  Numa base grande a maioria dos pontos cai nesse teste, e é ele que segura o
  custo: sem ele, todo ponto escreve na lista de tamanho `k`.
- **A inserção é ordenada em `K` posições**, `O(K)` por entrada e não `O(N)`.
  A lista é pequena e mantém-se ordenada, então o descarte antecipado é só
  uma comparação.
- **O laço de distância vai até `KNN_DIM − 1`**, pulando a última dimensão,
  que guarda o rótulo. Incluí-la daria um termo constante e errado na
  distância.

---

## 15. DWT — Discrete Wavelet Transform

### O que é

A DWT decompõe um sinal em **faixas de frequência**, como a FFT — mas com uma
propriedade que a FFT não tem: **as partes ficam em escalas diferentes**.

Uma FFT de 4096 pontos devolve 2048 coeficientes, todos na mesma resolução. Um
nível de DWT devolve duas metades do tamanho original:

- **a aproximação** `a[]` — a forma geral do sinal, ainda com resolution
  completa;
- **o detalhe** `d[]` — o que foi descartado nesse nível, já na metade da
  resolução.

A aproximação é então transformada de novo. Com 6 níveis, o sinal fica
decomposto em 6 faixas de detalhe mais uma aproximação final. É
**multirresolução**.

### Por que banco de filtros

Um nível é literalmente dois filtros FIR seguidos de subamostragem por 2:

```
a[i] = Σ_j lo[j] · x[2i + j]
d[i] = Σ_j hi[j] · x[2i + j]
```

O par `lo`/`hi` é um par **QMF** (Quadrature Mirror Filter), com a forma:

```
hi[j] = −(−1)^j · lo[L−1−j]
```

`L` é o comprimento: 2 para Haar, 4 para db4.

O detalhe útil dessa propriedade é que os filtros são **ortonormais**, e
isso dá **conservação de energia exata**:

```
Σ a_l² + Σ d_l² = Σ x_l²
```

Cada nível preserva a energia da própria entrada. O `energy_ratio` impresso no
relatório é essa propriedade medida, e vale para Haar e para db4 — sai entre
`0.999996` e `0.999999`, ou seja, 1 dentro do arredondamento de float32.

Uma armadilha clássica: ao somar os níveis, a energia é contada **uma vez
só** — guardam-se todos os detalhes `d₀…d_{L−1}` e **apenas a aproximação
final** `a_{L−1}`. Somar também `a₀, a₁, …` contaria a mesma energia mais de
uma vez, porque `a_l` é a entrada do nível `l+1`.

### Por que O(N)

Cada nível custa metade do anterior, então a soma é uma progressão
geométrica:

```
N/2 + N/4 + N/8 + … ≈ N
```

Com 4096 amostras e 6 níveis, o nível 6 processa 64 amostras — 1,6% do total.
É o motivo de o custo por nível **cair**, e não ficar plano.

### Onde é usado

- **Denoising de ECG e EEG** — o batimento cardíaco é o detalhe grosso
  (baixa frequência); o ruído muscular é detalhe fino. Descartar o detalhe fino
  limpa o sinal.
- **Deconvolução de sinais** — para obter `H = S/Y` é preciso estimar o
  espectro. Na DWT isso vira **divisão por faixa**, e é muito mais rápido que
  mínimos quadrados no tempo. É a aplicação clássica, de 1995.
- **Detecção de transientes e compressão** — em osciloscópio, defeito em
  rolamento, tremor de linha de produção: o defeito aparece como detalhe
  grudado em uma banda específica.
- **Features para classificação** — energia em cada sub-banda vira vetor de
  entrada para os classificadores daqui. É a ponte natural entre os itens
  11–14 e o 15.
- **Compressão** — os detalhes finos quase sempre são pequenos, então
  descarta-se ou quantiza-se sem perder o que importa.

### Onde o projeto mede

`main.c:dwt_one_level`. 4096 amostras, wavelet **Haar** (2 taps) e **db4**
(4 taps), 1 a 6 níveis.

Medida: **ciclos por transformada** em função do número de níveis, para as
duas wavelets.

### Notas de implementação

- **O índice é circular, sem módulo no laço.** `k = (2i + j) mod n`. Como
  `2i < n` e `j < L`, **uma única subtração** resolve o wrap. Módulo é divisão
  inteira, e o laço quente é esse.
- **A e `d` saem no mesmo passe.** Cálculos independentes, então é o mesmo
  laço, com o dobro de trabalho útil — metade das multiplicações seria
  desperdício de largura de banda de memória.
- **Os ponteiros de cada nível são calculados uma vez**, fora da região
  medida. A arena é dimensionada com a soma geométrica
  (`2·(N/2 + N/4 + …)`), não `6·N/2`: alocar a metade cheia seis vezes seria
  quase o dobro de RAM sem ganho.
- **A verificação é Parseval, não uma comparação com referência.** Para
  filtro ortonormal a razão de energia tem de dar 1, e um banco mal
  configurado entrega outro número imediatamente.
- **Haar e db4 são as duas ortonormais, e o custo é o número de taps.** Haar
  tem 2 taps e é a wavelet mais simples que existe — a diferença e a média de
  pares de amostras. db4 tem 4 taps, o que dobra o número de multiplicações por
  nível e dá uma separação de frequências muito mais nitida, com melhor
  compactação e melhor rejeição de sinal. Na prática o custo medido de db4 fica
  em torno de **1,67x** o de Haar, e não 2x, porque o laço interno mais longo
  amortiza melhor o custo de setup do laço. As duas preservam energia: os
  filtros `low` são ortonormais e o `high` é o par QMF,
  `hi[j] = −(−1)^j · lo[L−1−j]`, o que se confirma exatamente nos valores
  implementados.

---

## 16. DTW — Dynamic Time Warping

### O que é

Compara duas séries temporais permitindo que uma seja **esticada no tempo**.
Duas gravações da mesma frase nunca estão alinhadas no tempo: alguém fala mais
rápido, alguém demora mais no início. Comparar posição a posição dá uma
distância enorme para sinais que são na verdade idênticos.

O DTW encontra o **caminho de alinhamento** que minimiza o custo acumulado:

```
D[i][j] = c(i,j) + min( D[i−1][j], D[i][j−1], D[i−1][j−1] )
```

Com `c(i,j)` o custo local — aqui o quadrado da diferença euclidiana entre as
características. É programação dinâmica sobre a matriz de custos, e `D[n][m]`
no canto é a distância.

### O problema, e a janela que resolve

A matriz inteira custa `O(N·M)` **em tempo e em memória**. Com `N = M = 128`,
são 64 KB só para a matriz. Em borda, isso é a resposta automática.

A **janela Sakoe-Chiba** limita o caminho a `|i − j| ≤ r`: um descompasso de
tempo maior que `r` amostras é considerado improvável. Duas consequências:

**Tempo** cai de `O(N·M)` para `O(N·r)`.

**Memória** cai de `O(N·M)` para **`O(2r+1)`** — porque cada linha só toca
`2r+1` colunas, então **duas linhas bastam**, como no algoritmo de Kadane.

É essa a métrica que o documento pede: RAM contra a janela.

### O que a janela custa em qualidade

Restringir o caminho **pode aumentar a distância**. Se o alinhamento verdadeiro
precisa de `|i − j| = 20` e `r = 4`, o algoritmo é obrigado a pagar mais. Nos
resultados isso aparece como a distância **caindo** conforme `r` cresce e
depois **estabilizando** — o ponto em que a janela já contém o caminho
ótimo. Medir a distância junto do tempo mostra esse platô, e ele é a
evidência de que a janela escolhida é suficiente.

### Onde é usado

- **Reconhecimento de gestos por IMU** — a mesma sequência de aceleração para
  a pessoa fazer "círculo", rodada por rodada. É a aplicação mais clássica em
  borda, e combina direto com o k-NN do item 14.
- **Comparação de batimentos de ECG** — identifica um batimento numa gravação
  longa, com distorções de tempo.
- **Alinhamento de fala** — encontrar a mesma palavra falada por pessoas
  diferentes ou na mesma pessoa em gravações distintas.
- **Recuperação de música** — casar um trecho tocado com outro, em outra
  andamento.
- **Sensores com deriva** — comparar o padrão atual com o padrão gravado em
  outra velocidade de amostragem.

### Quando não usar

- **Dois sinais com descompasso grande e conhecido** — aí o problema é
  *time warping* **global**, e uma inclinação reta resolve por muito menos
  (`DTW` resolve descompasso **local**, que é o motivo de ser caro).
- **Sinal contínuo e deslizante** — recalcular o DTW a cada amostra é
  proibitivo; dá para usar o **caixa de Marguerite** (SB-DTW) ou calcular só
  numa janela.
- **Streaming com muitas classes** — `N_sequências` comparações por
  inferência.

### Onde o projeto mede

`main.c:dtw_run`. Duas séries de 128 amostras × 8 dimensões, janela
`r = 4 … 64`.

Medida: **µs por comparação** e **bytes de RAM** em função do raio `r`.

### Notas de implementação

- **O mapeamento coluna → posição na linha é `p = j − i + r + 1`.** Como `p`
  desloca exatamente 1 a cada linha, duas linhas de `2r+3` floats bastam e
  nenhuma precisa ser redimensionada. Os slots 0 e `2r+2` são as guardas, e
  ficam em infinito.
- **As três leituras do recurrence** caem em posições fixas:
  `D[i−1][j] = prev[p+1]`, `D[i−1][j−1] = prev[p]`, `D[i][j−1] = cur[p−1]`.
- **A linha inteira é posto em infinito a cada iteração**, e não só a faixa
  válida. Sem isso, os slots que a janela de cima não alcança ficariam com
  valor velho de **duas linhas atrás** e aceitariam um caminho que a janela
  proíbe. O custo é `O(2r+1)` por linha, a mesma ordem do laço de custo.
- **A linha virtual −1 tem de ser inicializada em infinito.** É ela que obriga
  o caminho a começar em `(0,0)`. Sem essa inicialização, o buffer começa com
  lixo de heap e a primeira linha aceita partir de um custo inválido, o que
  **abaixa a distância sem motivo** — e não dá erro, só número errado.
- **A propagação na linha 0 e na coluna 0 é correta** e foi mantida: no DTW
  padrão, `D[0][j]` e `D[i][0]` existem. O infinito fica no índice `−1`, não
  em `0`.
- **A implementação foi validada contra uma referência** de matriz completa
  com a mesma janela, em 500 pares aleatórios × 11 raios: **erro relativo
  0.000e+00** em todos os casos, inclusive com os buffers envenenados com
  `0xA5` para provar a inicialização em infinito.

---

## Comparando os seis

| Algoritmo | Custo | Treina? | Memória do modelo | Robusto a outliers |
|-----------|-------|---------|-------------------|--------------------|
| LMS | médio, `O(N·M)` | sim (online) | `O(M)` | não é classificação |
| SVM linear | baixo, `O(D)` reduzido | sim (offline) | `O(D)` reduzido | com kernel RBF |
| SVM RBF | alto, `O(D·SV)` + `exp` | sim (offline) | `O(D·SV)` | sim |
| Random Forest | médio, ~`O(T·prof)` | sim (offline) | `O(T·prof)`, o mais gordo | sim |
| k-NN | **alto**, `O(N·D)` | não | `O(N·D)`, o maior | com `k` maior |
| DWT | baixo, `O(N)` | não | `O(N)` | não é classificação |
| DTW | **alto**, `O(N·r)` | não | `O(2r+1)` | não é classificação |

Três leituras que os números da placa devem confirmar:

1. **SVM linear reduzido é o mais barato da lista.** Colapsar os
   vetores-suporte em `w` tira o custo de `O(D·SV)` e libera a RAM da base. É
   a otimização de maior impacto de todo o documento para borda.
2. **k-NN é o mais caro e o mais guloso.** Não treina, mas paga o preço
   inteiro em tempo de inferência e em RAM a cada uso.
3. **DTW e k-NN juntos são o pipeline natural de gestos:** DTW alinha, k-NN
   classifica. E é por isso que os dois últimos itens do documento são esses.

---

## Por que medir no ESP32-S3

O custo em ciclos no desktop é quase irrelevante para o projeto de borda. No
ESP32-S3 o que importa é:

- **O custo é o orçamento de tempo.** Para áudio a 16 kHz, uma amostra a cada
  62,5 µs. Se o filtro passar disso, o sistema atrasa.
- **RAM é o orçamento de memória.** Buffers grandes consomem SRAM, que é
  escasso. O Goertzel (item 8) usar `O(1)` em vez de `O(N)` pode ser a
  diferença entre caber e não caber — e o mesmo vale para a base do k-NN e para
  os vetores-suporte do SVM.
- **Custo de RAM é custo de tempo, no microcontrolador.** O processador é rápido
  e a memória é lenta; ler 52 KB da base do k-NN custa **mais** que as
  multiplicações. Por isso o `k`-NN é tão caro aqui, e por isso o descarte
  antecipado do item 14 importa tanto.
- **Determinismo.** Na borda, a latência previsível importa tanto quanto a
  média.

---

# Parte 2 — Como usar o projeto

## Por que em C puro

O documento cita **micromlgen** e **emlearn** para SVM (item 12) e Random Forest
(item 13). Nenhuma das duas está disponível como componente do ESP-IDF — o
emlearn é Arduino/C++, e o esp-dsp (que tem uma SVM pronta) também é C++. O
padrão dos projetos anteriores, mantido aqui, é **implementar em C puro** para
ficar autocontido e comparável com os demais itens.

Além disso, **nenhum destes seis itens usa o esp-dsp** — não há FFT nem banco
de filtros dele. A dependência `espressif/esp-dsp` foi omitida de propósito
deste projeto, ao contrário dos dois anteriores.

Se a intenção for comparar com o emlearn de verdade, o caminho é gerar o C com
ele no PC e colar o resultado como um `main/xxx.c` — é exatamente a forma como
o emlearn é usado fora do Arduino.

## Ambiente

ESP-IDF 5.5 ou 6.x e uma placa ESP32-S3. Compilado e validado com
**ESP-IDF v6.0.2** para o alvo `esp32s3`.

## Compilar e gravar

```bash
cd esp32s3_benchmark_6_algoritmos_3

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
CSV,algoritmo,parametro,ciclos,tempo_ns,checksum
```

Exemplo do formato (os **números abaixo são ilustrativos** — os valores reais
saem da placa):

```text
CSV,LMS,8,...,...,...          # param = número de taps
CSV,SVM,8,...,...,...          # param = vetores-suporte, kernel linear
CSV,SVMRBF,8,...,...,...       # param = vetores-suporte, kernel RBF
CSV,RFOREST,8,...,...,...      # param = número de árvores
CSV,KNN,64,...,...,...         # param = tamanho da base
CSV,DWT_HAAR,1,...,...,...     # param = número de níveis
CSV,DWT_DB4,1,...,...,...      # param = número de níveis
CSV,DTW,4,...,...,...          # param = raio da janela
```

O que a coluna de tempo significa **depende do algoritmo**, e é preciso saber:

| Algoritmo | `ciclos` | `tempo_ns` |
|-----------|----------|------------|
| LMS | por **amostra** (normalizado) | ns por amostra |
| SVM / SVMRBF | por **inferência** | µs por inferência |
| RFOREST | por **inferência** | µs por inferência |
| KNN | por **inferência** | µs por inferência |
| DWT_* | por **transformada** (absoluto) | µs por transformada |
| DTW | por **comparação** (absoluto) | µs por comparação |

O checksum existe para impedir que o compilador elimine o cálculo por ser
aparentemente inútil.

Além do CSV, cada algoritmo imprime uma linha de resumo com as métricas
secundárias, e é aí que ficam as **verificações numéricas**:

```text
LMS_RESULT,taps=32,cycles_per_sample=...,ns_per_sample=...,
       residual_mse_db=...,iters_to_20db=...,weight_rmse=...
SVM_RESULT,n_sv=64,us_per_inference=...,cycles_per_inference=...,
       us_collapsed=...,dual_vs_collapsed_diff=...,accuracy=...
SVMRBF_RESULT,n_sv=64,us_per_inference=...,cycles_per_inference=...,
       gamma=...,accuracy=...
RFOREST_RESULT,trees=32,us_per_inference=...,cycles_per_inference=...,
       nodes=...,nodes_per_tree=...,model_bytes=...,model_kib=...,
       depth=...,accuracy=...,min_free_heap=...
KNN_RESULT,db_size=512,k=3,us_per_inference=...,cycles_per_inference=...,
      accuracy=...,base_bytes=...,base_kib=...
DWT_HAAR_RESULT,levels=3,samples=4096,cycles=...,
           cycles_per_sample=...,energy_ratio=...
DTW_RESULT,band=16,slots_per_row=...,us_per_comparison=...,
        cycles_per_comparison=...,ram_bytes=...,ram_kib=...,distance=...
```

### O que conferir nas linhas de resumo

Estes campos são as verificações de correção, e devem ter estes valores:

| Campo | Valor esperado | O que prova |
|-------|----------------|-------------|
| `weight_rmse` (LMS) | `10⁻⁸` a `10⁻⁷` | o NLMS convergiu para o sistema real |
| `iters_to_20db` (LMS) | cresce com `M` | taxa de convergência ~`O(M)` |
| `dual_vs_collapsed_diff` (SVM) | `3·10⁻⁸` a `7·10⁻⁷` | a forma reduzida é a mesma função |
| `accuracy` (SVM/RF/kNN) | alto | a implementação classifica certo |
| `energy_ratio` (DWT) | `0.99999`–`1.00000` | Parseval: filtros ortonormais |
| `distance` (DTW) | cai e estabiliza | a janela `r` já contém o caminho ótimo |

O `energy_ratio` em particular é a verificação mais forte do projeto: se a
convenção do banco de filtros ou os coeficientes da wavelet estivessem
errados, ele sairia diferente de 1 **imediatamente**.

## Repetições

Por padrão são feitas 30 execuções de cada configuração. As 5 primeiras são
descartadas como warm-up.

Altere em `main/main.c`:

```c
#define BENCH_REPS 30
#define WARMUP_REPS 5
```

O benchmark usa **média**. Para um relatório mais rigoroso, vale registrar
também mínimo e desvio — o material de `esp32s3_benchmark_5_algoritmos_2`
descreve essa prática.

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

O script usa o backend `Agg`, então roda sem display e salva 8 PNGs
(`lms_ciclos.png`, `svm_ciclos.png`, `svmrbf_ciclos.png`,
`rforest_ciclos.png`, `knn_ciclos.png`, `dwt_haar_ciclos.png`,
`dwt_db4_ciclos.png`, `dtw_ciclos.png`) mais `resultados_resumido.csv`.

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

Cuidado específico deste projeto: **cuidado com a RAM nos classificadores**. A
base do k-NN (52 KB com 1024 vetores) e os nós da Random Forest ocupam SRAM
junto com o resto. Por isso o `RFOREST_RESULT` imprime `min_free_heap`: se a
floresta grande não couber, o laço avisa e segue para a próxima configuração em
vez de quebrar.

## O que você deve colocar no relatório

Para cada algoritmo, gere uma tabela e um gráfico:

- **LMS**: eixo X número de taps, eixo Y ciclos por amostra; mais a curva de
  convergência (`iters_to_20db` e `weight_rmse`);
- **SVM linear**: eixo X vetores-suporte, eixo Y µs por inferência; comparar
  com `us_collapsed` para mostrar o ganho da forma reduzida;
- **SVM RBF**: eixo X vetores-suporte, eixo Y µs por inferência; comparar com
  o linear para isolar o custo do `expf`;
- **Random Forest**: eixo X número de árvores, eixo Y µs por inferência; e
  **µs × KiB de modelo**, que é a curva que mostra o preço da acurácia
  extra;
- **k-NN**: eixo X tamanho da base, eixo Y µs por inferência;
- **DWT**: eixo X número de níveis, eixo Y ciclos por transformada, uma curva
  para Haar e uma para db4;
- **DTW**: eixo X raio da janela, eixo Y µs por comparação; e um segundo
  gráfico de **RAM contra raio**.

O documento original informa as complexidades e métricas sugeridas; os
resultados reais devem ser obtidos na placa.

## Referências

- Lyons, *Understanding Digital Signal Processing*
- Proakis & Manolakis, *Digital Signal Processing: Principles, Algorithms and
  Applications*
- Haykin, *Adaptive Filter Theory*
- Widrow & Hoff, *Adaptive Switching Circuits* (1960) — o LMS original
- Cortes & Vapnik, *A Support-Vector Network* (1995)
- Sakoe & Chiba, *Dynamic Programming Algorithm Optimization for the Warping
  Problem* (1978) — a janela que dá nome ao método
- Breiman, *Random Forests* (2001)
- Daubechies, *Ten Lectures on Wavelets*
- Mallat, *A Wavelet Tour of Signal Processing*
- Espressif, *ESP-DSP Programming Guide* — usado nos projetos 1 e 2, não
  neste

## Fim da série

Com este projeto, os 16 itens do documento estão implementados:

| Projeto | Itens | Algoritmos |
|---------|-------|------------|
| `esp32s3_benchmark_5_algoritmos` | 1–5 | FFT, FIR, CNN, IIR biquad, MLP |
| `esp32s3_benchmark_5_algoritmos_2` | 6–10 | MFCC, Kalman, Goertzel, mediana, média móvel |
| `esp32s3_benchmark_6_algoritmos_3` | 11–16 | LMS, SVM, Random Forest, k-NN, DWT, DTW |

Vale comparar as três séries: **o custo em ciclos por operação é da ordem de
1 a 10** para um operador simples (média móvel, Goertzel) e da ordem de
**centenas** para um operador com laços aninhados ou função transcendental
(RBF, árvore). E o número de níveis do DWT mostra que **metade do custo de um
algoritmo pode estar no detalhe fino do parâmetro**.
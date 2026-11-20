# Projeto 01 — Implementação Paralela do Algoritmo K-means (Sequencial, OpenMP e CUDA)

**Disciplina:** Computação Paralela — ICEI / PUC Minas  
**Linguagens / Tecnologias:** C (C99), OpenMP (CPU Multicore), CUDA (GPU NVIDIA)

---

## 1. Visão Geral do Projeto

Este repositório contém a implementação do algoritmo de agrupamento (*clustering*) **K-means** em linguagem C e suas respectivas paralelizações voltadas para duas arquiteturas distintas:

1. **Versão Sequencial (`kmeans_seq.c`):** Implementação de referência (*ground truth*) utilizada para validação de corretude e linha de base ($T_{\text{seq}}$) para o cálculo de *speedup*.
2. **Versão OpenMP (`kmeans_omp.c`):** Paralelização para arquiteturas *multicore* de memória compartilhada (CPU), avaliada com escalonamento de $1, 2, 4, 8, 16$ e $32$ threads.
3. **Versão CUDA (`kmeans_cuda.cu`):** Paralelização massiva para arquiteturas heterogêneas com GPU NVIDIA (modelo *Host/Device*).

### Funcionamento do Algoritmo e Padrões Paralelos

Dado um conjunto de $N$ pontos em um espaço de $D$ dimensões, o algoritmo particiona os dados em $K$ clusters repetindo duas etapas até atingir convergência exata (nenhum ponto muda de grupo) ou o limite máximo de iterações:

* **Etapa 1 — Atribuição (*Assignment* — Padrão *Map*):** Calcula a distância euclidiana ao quadrado de cada ponto $p_i$ até os $K$ centróides, associando-o ao mais próximo. Como cada ponto é independente, a etapa é massivamente paralelizável.
* **Etapa 2 — Atualização (*Update* — Padrão *Reduce*):** Recalcula cada centróide como a média aritmética das coordenadas dos pontos atribuídos ao seu grupo. Em paralelo, utiliza acumuladores privados por thread (OpenMP) e operações atômicas/memória compartilhada (CUDA) para evitar condições de corrida.

---

## 2. Estrutura do Repositório

```text
CoPa_TP1/
├── Base_de_dados/
│   ├── YearPredictionMSD.txt     # Dataset real principal (N=515345, D=90)
│   └── teste_5k.txt              # Subconjunto leve para testes rápidos de validação
├── kmeans_sequencial/
│   ├── kmeans_seq.c              # Implementação sequencial de referência
│   └── executar.sh               # Script automatizado de compilação, benchmark e diff
├── kmeans_openmp/
│   └── kmeans_omp.c              # Implementação paralela CPU (1 a 32 threads)
├── kmeans_cuda/
│   └── kmeans_cuda.cu            # Implementação paralela GPU (CUDA)
└── README.md                     # Documentação geral do projeto
```

*(Nota: Caso mantenha todos os arquivos `.c`, `.cu` e `executar.sh` no mesmo diretório para empacotamento final no `.zip`, o script `executar.sh` detecta, compila e valida automaticamente todas as versões presentes).*

---

## 3. Bases de Dados Utilizadas

### 3.1 Base Oficial de Benchmark: *YearPredictionMSD* (UCI ML Repository)
* **Origem:** *Million Song Dataset* (UCI Machine Learning Repository)
* **Dimensões:** $N = 515.345$ instâncias $\times$ $D = 90$ atributos contínuos de timbre de áudio (`float`).
* **Formato:** Arquivo texto separado por vírgulas (`.txt`), sem cabeçalho. A coluna $0$ contém o ano da música e é descartada automaticamente na leitura (`skip_col0 = 1`).
* **Calibragem:** Com $K = 64$ clusters e $50$ iterações, a versão sequencial demanda mais de $10$ segundos de computação pura, atendendo ao requisito mínimo do trabalho.

### 3.2 Base Rápida para Testes de Desenvolvimento
Para validar alterações rapidamente sem aguardar a execução completa:
```bash
head -n 5000 Base_de_dados/YearPredictionMSD.txt > Base_de_dados/teste_5k.txt
```

---

## 4. Requisitos de Ambiente

* **Compilador C com suporte a OpenMP:** `gcc` (ou `gcc-8` no servidor PARCODE)
* **Toolkit NVIDIA CUDA (para versão GPU):** `nvcc` e GPU compatível (ex.: NVIDIA GeForce GT 1030 no PARCODE)
* **Sistema Operacional:** Linux (Ubuntu / ambiente ICEI / PARCODE)

Verifique o suporte no terminal antes de compilar:
```bash
gcc --version
echo | cpp -fopenmp -dM | grep -i openmp
nvcc --version
```

---

## 5. Compilação e Execução

### 5.1 Execução Automatizada (`executar.sh`)
Dentro da pasta onde estão os códigos-fonte, conceda permissão de execução e rode o script apontando para a base de dados:

```bash
chmod +x executar.sh

# Benchmark oficial completo (YearPredictionMSD: N=515345, D=90, K=64, max_iter=50, skip_col0=1)
./executar.sh ../Base_de_dados/YearPredictionMSD.txt

# Teste rápido de validação (teste_5k.txt: N=5000, D=90, K=8, max_iter=20, skip_col0=1)
./executar.sh ../Base_de_dados/teste_5k.txt 5000 90 8 20 1
```

O script compila `kmeans_seq.c`, executa `kmeans_omp.c` para **$1, 2, 4, 8, 16$ e $32$ threads**, executa `kmeans_cuda.cu` e verifica a equivalência dos resultados contra `centroides_seq.txt` usando `diff`.

### 5.2 Compilação e Execução Manual

#### Versão Sequencial
```bash
gcc -O3 -std=c99 -Wall -Wextra kmeans_seq.c -o kmeans_seq
./kmeans_seq ../Base_de_dados/YearPredictionMSD.txt 515345 90 64 50 1 centroides_seq.txt
```

#### Versão OpenMP (CPU Multicore)
```bash
gcc -O3 -std=c99 -Wall -Wextra -fopenmp kmeans_omp.c -o kmeans_omp

# Exemplo rodando com 4 threads:
OMP_NUM_THREADS=4 ./kmeans_omp ../Base_de_dados/YearPredictionMSD.txt 515345 90 64 50 1 centroides_omp_4.txt
```

#### Versão CUDA (GPU)
```bash
nvcc -O3 kmeans_cuda.cu -o kmeans_cuda
./kmeans_cuda ../Base_de_dados/YearPredictionMSD.txt 515345 90 64 50 1 centroides_cuda.txt
```

### 5.3 Assinatura de Linha de Comando (Padrão Comum)
Todos os executáveis seguem a mesma interface:
```text
./<executavel> <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0> [arquivo_saida.txt]
```

---

## 6. Decisões de Arquitetura e Otimização

* **Layout de Memória Contíguo (1D):** Todas as matrizes (`pontos[N * D]`, `centroides[K * D]`, `somas[K * D]`) são alocadas como vetores lineares contíguos em `float` (32 bits), maximizando o aproveitamento de cache na CPU, garantindo acesso coalescido na GPU e permitindo transferência única (`cudaMemcpyHostToDevice`) antes do laço iterativo.
* **Remoção de Raiz Quadrada (`sqrtf`):** Como $f(x) = \sqrt{x}$ é estritamente crescente para $x \ge 0$, as atribuições comparam diretamente a distância euclidiana ao quadrado ($\sum (p_d - c_d)^2$), eliminando instruções de alto custo no laço interno.
* **Inicialização Determinística:** Os primeiros $K$ pontos do dataset são escolhidos como centróides iniciais em todas as versões, assegurando trajetória de convergência idêntica.
* **Isolamento de E/S na Cronometria:** O tempo de leitura de disco (`Tempo de I/O`) e a escrita de validação são medidos separadamente do laço computacional (`Tempo K-means`).

---

## 7. Resultados Experimentais e Análise de Escalabilidade

*(Preencher a tabela abaixo após executar os benchmarks finais na mesma máquina de referência).*

* **Ambiente de Teste:** `[Ex: Servidor PARCODE — Intel Core i5 (4 núcleos) + NVIDIA GT 1030 (384 cores)]`
* **Parâmetros:** `YearPredictionMSD` ($N = 515.345$, $D = 90$, $K = 64$, `max_iter` $= 50$)

| Implementação / Configuração | Tempo K-means (s) | Speedup ($S = T_{\text{seq}} / T_p$) | SSE Final | Validação (`diff`) |
| :--- | :---: | :---: | :---: | :---: |
| **Sequencial (`kmeans_seq`)** | `[XX.XXXXXX]` | `1.00x` | `[X.XXXXe+XX]` | Referência |
| **OpenMP — 1 thread** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **OpenMP — 2 threads** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **OpenMP — 4 threads** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **OpenMP — 8 threads** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **OpenMP — 16 threads** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **OpenMP — 32 threads** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |
| **CUDA (`kmeans_cuda`)** | `[XX.XXXXXX]` | `[X.XXx]` | `[X.XXXXe+XX]` | OK |

### Observações sobre Desempenho
1. **Escalabilidade em CPU (OpenMP):** O ganho de desempenho acompanha o número de núcleos físicos da CPU. Ao atingir $16$ e $32$ threads (acima da capacidade física do processador), ocorre *oversubscription*, aumentando a contenção de memória e trocas de contexto.
2. **Aceleração em GPU (CUDA):** A cópia única da matriz de pontos ($N \times D$) da RAM para a VRAM antes do laço principal amortiza o custo do barramento PCIe ao longo das iterações, permitindo que milhares de threads processem os pontos simultaneamente.

---

## 8. Equipe

* **Integrante 1:** Versão Base Sequencial (`kmeans_seq.c`), Paralelização OpenMP (`kmeans_omp.c`) e Preparação da Base de Dados.
* **Integrante 2:** Paralelização GPU (`kmeans_cuda.cu`), Profiling CUDA e Automação de Testes.
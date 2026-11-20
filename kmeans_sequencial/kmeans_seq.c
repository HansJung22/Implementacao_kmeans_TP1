/*
 * ============================================================================
 * Projeto 01 - Implementação Paralela do Algoritmo K-means (Computação Paralela)
 * Arquivo: kmeans_seq.c (Versão Sequencial de Referência)
 * ============================================================================
 *
 * TEMPOS DE EXECUÇÃO (Preencher após benchmarks oficiais na máquina de teste):
 *   Hardware : [Ex: PARCODE - Intel Core / PC Pessoal]
 *   Dataset  : [Ex: YearPredictionMSD (N=515345, D=90, K=64, max_iter=50)]
 *   Sequencial (1 thread): [XX.XXXXXX s]
 *
 * COMPILAÇÃO:
 *   gcc -O3 -std=c99 -Wall -Wextra kmeans_seq.c -o kmeans_seq
 *
 * EXECUÇÃO:
 *   ./kmeans_seq <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0> [saida.txt]
 *
 * EXEMPLOS:
 *   YearPredictionMSD (descarta coluna 0 = ano):
 *     ./kmeans_seq YearPredictionMSD.txt 515345 90 64 50 1 centroides_seq.txt
 *
 *   Covertype (usa todas as 54 colunas numéricas):
 *     ./kmeans_seq covtype.data 581012 54 64 50 0 centroides_seq.txt
 *
 *   Arquivo pré-convertido em binário (extensão .bin):
 *     ./kmeans_seq dados.bin 515345 90 64 50 0 centroides_seq.txt
 * ============================================================================
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* Estrutura que agrupa todos os vetores 1D contíguos do K-means */
typedef struct {
    int N;               /* Número de pontos */
    int D;               /* Número de dimensões por ponto */
    int K;               /* Número de clusters (centróides) */
    float *pontos;       /* Vetor 1D [N * D]: pontos[i * D + d] */
    float *centroides;   /* Vetor 1D [K * D]: centroides[c * D + d] */
    float *somas;        /* Vetor 1D [K * D]: somatório auxiliar da atualização */
    int   *cluster_assoc;/* Vetor 1D [N]: índice do cluster de cada ponto */
    int   *contagens;    /* Vetor 1D [K]: quantidade de pontos por cluster */
} KMeansData;

/* ----------------------------------------------------------------------------
 * Cronometria monotônica de alta precisão (segundos)
 * ---------------------------------------------------------------------------- */
static double tempo_atual_seg(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ----------------------------------------------------------------------------
 * Alocação e liberação dos vetores 1D contíguos
 * ---------------------------------------------------------------------------- */
static int alocar_kmeans(KMeansData *km, int N, int D, int K) {
    km->N = N;
    km->D = D;
    km->K = K;

    km->pontos        = (float *)malloc((size_t)N * (size_t)D * sizeof(float));
    km->centroides    = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    km->somas         = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    km->cluster_assoc = (int *)malloc((size_t)N * sizeof(int));
    km->contagens     = (int *)malloc((size_t)K * sizeof(int));

    if (!km->pontos || !km->centroides || !km->somas ||
        !km->cluster_assoc || !km->contagens) {
        fprintf(stderr, "Erro: falha ao alocar memória para N=%d, D=%d, K=%d.\n", N, D, K);
        return -1;
    }
    return 0;
}

static void liberar_kmeans(KMeansData *km) {
    free(km->pontos);
    free(km->centroides);
    free(km->somas);
    free(km->cluster_assoc);
    free(km->contagens);
}

/* ----------------------------------------------------------------------------
 * Leitura da base de dados (suporta CSV numérico e formato binário .bin)
 * ---------------------------------------------------------------------------- */
static int termina_com(const char *str, const char *sufixo) {
    size_t len_str = strlen(str);
    size_t len_suf = strlen(sufixo);
    if (len_suf > len_str) return 0;
    return strcmp(str + len_str - len_suf, sufixo) == 0;
}

static int carregar_binario(const char *caminho, KMeansData *km) {
    FILE *fp = fopen(caminho, "rb");
    if (!fp) {
        perror("Erro ao abrir arquivo binário");
        return -1;
    }

    int n_arq = 0, d_arq = 0;
    if (fread(&n_arq, sizeof(int), 1, fp) != 1 ||
        fread(&d_arq, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Erro ao ler cabeçalho (N, D) do arquivo binário.\n");
        fclose(fp);
        return -1;
    }

    if (n_arq < km->N || d_arq != km->D) {
        fprintf(stderr, "Erro: dimensões do binário (%d x %d) incompatíveis com (%d x %d).\n",
                n_arq, d_arq, km->N, km->D);
        fclose(fp);
        return -1;
    }

    size_t total_floats = (size_t)km->N * (size_t)km->D;
    size_t lidos = fread(km->pontos, sizeof(float), total_floats, fp);
    fclose(fp);

    if (lidos != total_floats) {
        fprintf(stderr, "Erro: leitura incompleta (%zu de %zu floats).\n", lidos, total_floats);
        return -1;
    }
    return 0;
}

static int carregar_csv(const char *caminho, KMeansData *km, int skip_col0) {
    FILE *fp = fopen(caminho, "r");
    if (!fp) {
        perror("Erro ao abrir arquivo CSV");
        return -1;
    }

    char linha[16384];
    int i = 0;

    while (i < km->N && fgets(linha, sizeof(linha), fp)) {
        char *ptr = linha;
        char *fim = NULL;

        /* Descarta a primeira coluna se solicitado (ex: ano em YearPredictionMSD) */
        if (skip_col0) {
            (void)strtof(ptr, &fim);
            ptr = (*fim == ',' || *fim == ';' || *fim == '\t') ? fim + 1 : fim;
        }

        float *destino = &km->pontos[(size_t)i * (size_t)km->D];
        for (int d = 0; d < km->D; d++) {
            destino[d] = strtof(ptr, &fim);
            if (fim == ptr) {
                fprintf(stderr, "Erro de parsing na linha %d, coluna %d.\n", i + 1, d);
                fclose(fp);
                return -1;
            }
            ptr = (*fim == ',' || *fim == ';' || *fim == '\t') ? fim + 1 : fim;
        }
        i++;
    }

    fclose(fp);
    if (i < km->N) {
        fprintf(stderr, "Erro: arquivo contém apenas %d registros (esperado %d).\n", i, km->N);
        return -1;
    }
    return 0;
}

static int carregar_dados(const char *caminho, KMeansData *km, int skip_col0) {
    if (termina_com(caminho, ".bin")) {
        return carregar_binario(caminho, km);
    }
    return carregar_csv(caminho, km, skip_col0);
}

/* ----------------------------------------------------------------------------
 * Inicialização determinística (primeiros K pontos da base)
 * ---------------------------------------------------------------------------- */
static void inicializar_centroides(KMeansData *km) {
    size_t bytes_centroides = (size_t)km->K * (size_t)km->D * sizeof(float);
    memcpy(km->centroides, km->pontos, bytes_centroides);

    for (int i = 0; i < km->N; i++) {
        km->cluster_assoc[i] = -1;
    }
}

/* ----------------------------------------------------------------------------
 * ETAPA 1: ATRIBUIÇÃO (Padrão MAP)
 * Associa cada ponto ao centróide de menor distância euclidiana ao quadrado.
 * Retorna a quantidade de pontos que trocaram de cluster nesta iteração.
 * ---------------------------------------------------------------------------- */
static int etapa_atribuicao(KMeansData *km) {
    const int N = km->N;
    const int D = km->D;
    const int K = km->K;
    const float *pontos = km->pontos;
    const float *centroides = km->centroides;
    int *cluster_assoc = km->cluster_assoc;

    int mudancas = 0;

    for (int i = 0; i < N; i++) {
        const float *p = &pontos[(size_t)i * (size_t)D];
        float menor_dist = FLT_MAX;
        int melhor_c = 0;

        for (int c = 0; c < K; c++) {
            const float *cent = &centroides[(size_t)c * (size_t)D];
            float dist_sq = 0.0f;

            /* Distância euclidiana ao quadrado (sem sqrtf: preserva a ordenação) */
            for (int d = 0; d < D; d++) {
                float diff = p[d] - cent[d];
                dist_sq += diff * diff;
            }

            if (dist_sq < menor_dist) {
                menor_dist = dist_sq;
                melhor_c = c;
            }
        }

        if (cluster_assoc[i] != melhor_c) {
            cluster_assoc[i] = melhor_c;
            mudancas++;
        }
    }

    return mudancas;
}

/* ----------------------------------------------------------------------------
 * ETAPA 2: ATUALIZAÇÃO DOS CENTRÓIDES (Padrão REDUCE)
 * Soma as coordenadas dos pontos por cluster e recalcula a média aritmética.
 * ---------------------------------------------------------------------------- */
static void etapa_atualizacao(KMeansData *km) {
    const int N = km->N;
    const int D = km->D;
    const int K = km->K;
    const float *pontos = km->pontos;
    const int *cluster_assoc = km->cluster_assoc;
    float *somas = km->somas;
    float *centroides = km->centroides;
    int *contagens = km->contagens;

    /* Zera acumuladores da iteração */
    memset(somas, 0, (size_t)K * (size_t)D * sizeof(float));
    memset(contagens, 0, (size_t)K * sizeof(int));

    /* Acumulação das coordenadas e contagens */
    for (int i = 0; i < N; i++) {
        int c = cluster_assoc[i];
        contagens[c]++;

        float *soma_c = &somas[(size_t)c * (size_t)D];
        const float *p = &pontos[(size_t)i * (size_t)D];

        for (int d = 0; d < D; d++) {
            soma_c[d] += p[d];
        }
    }

    /* Normalização: divide soma pela contagem (mantém centróide se cluster vazio) */
    for (int c = 0; c < K; c++) {
        if (contagens[c] > 0) {
            float inv_count = 1.0f / (float)contagens[c];
            float *cent = &centroides[(size_t)c * (size_t)D];
            const float *soma_c = &somas[(size_t)c * (size_t)D];

            for (int d = 0; d < D; d++) {
                cent[d] = soma_c[d] * inv_count;
            }
        }
    }
}

/* ----------------------------------------------------------------------------
 * Validação pós-execução (FORA da cronometria): cálculo do SSE e exportação
 * ---------------------------------------------------------------------------- */
static double calcular_sse(const KMeansData *km) {
    double sse = 0.0;
    for (int i = 0; i < km->N; i++) {
        int c = km->cluster_assoc[i];
        const float *p = &km->pontos[(size_t)i * (size_t)km->D];
        const float *cent = &km->centroides[(size_t)c * (size_t)km->D];
        double dist_sq = 0.0;

        for (int d = 0; d < km->D; d++) {
            double diff = (double)p[d] - (double)cent[d];
            dist_sq += diff * diff;
        }
        sse += dist_sq;
    }
    return sse;
}

static void salvar_centroides(const char *caminho, const KMeansData *km) {
    FILE *fp = fopen(caminho, "w");
    if (!fp) {
        perror("Aviso: não foi possível salvar arquivo de centróides");
        return;
    }

    for (int c = 0; c < km->K; c++) {
        fprintf(fp, "Cluster %3d (n=%7d):", c, km->contagens[c]);
        const float *cent = &km->centroides[(size_t)c * (size_t)km->D];
        for (int d = 0; d < km->D; d++) {
            fprintf(fp, " %.4f", cent[d]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* ----------------------------------------------------------------------------
 * Função Principal
 * ---------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "Uso: %s <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0> [saida.txt]\n",
                argv[0]);
        return 1;
    }

    const char *arquivo_entrada = argv[1];
    int N         = atoi(argv[2]);
    int D         = atoi(argv[3]);
    int K         = atoi(argv[4]);
    int max_iter  = atoi(argv[5]);
    int skip_col0 = atoi(argv[6]);
    const char *arquivo_saida = (argc >= 8) ? argv[7] : "centroides_seq.txt";

    if (N <= 0 || D <= 0 || K <= 0 || K > N || max_iter <= 0) {
        fprintf(stderr, "Erro: parâmetros inválidos (verifique N, D, K <= N e max_iter).\n");
        return 1;
    }

    KMeansData km;
    if (alocar_kmeans(&km, N, D, K) != 0) {
        liberar_kmeans(&km);
        return 1;
    }

    /* Leitura de disco isolada da medição de desempenho do K-means */
    double t_io_ini = tempo_atual_seg();
    if (carregar_dados(arquivo_entrada, &km, skip_col0) != 0) {
        liberar_kmeans(&km);
        return 1;
    }
    double t_io_fim = tempo_atual_seg();

    /* Inicialização determinística antes do laço */
    inicializar_centroides(&km);

    /* ========================================================================
     * INÍCIO DA REGIÃO CRONOMETRADA (COMPUTAÇÃO K-MEANS)
     * ======================================================================== */
    double t_ini = tempo_atual_seg();

    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        int mudancas = etapa_atribuicao(&km);
        etapa_atualizacao(&km);

        if (mudancas == 0) {
            iter++;
            break;
        }
    }

    double t_fim = tempo_atual_seg();
    /* ========================================================================
     * FIM DA REGIÃO CRONOMETRADA
     * ======================================================================== */

    /* Métricas de desempenho e validação */
    double sse = calcular_sse(&km);
    salvar_centroides(arquivo_saida, &km);

    printf("=== K-means Sequencial ===\n");
    printf("Entrada          : %s (N=%d, D=%d, K=%d)\n", arquivo_entrada, N, D, K);
    printf("Iteracoes        : %d / %d\n", iter, max_iter);
    printf("Tempo de I/O     : %.6f s\n", t_io_fim - t_io_ini);
    printf("Tempo K-means    : %.6f s\n", t_fim - t_ini);
    printf("SSE final        : %.4e\n", sse);
    printf("Centroides salvos: %s\n", arquivo_saida);

    liberar_kmeans(&km);
    return 0;
}
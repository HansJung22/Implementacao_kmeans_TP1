/*
 * ============================================================================
 * Projeto 01 - Implementação Paralela do Algoritmo K-means (Computação Paralela)
 * Arquivo: kmeans_cuda.cu (Versão Paralela com CUDA)
 * ============================================================================
 *
 * TEMPOS DE EXECUÇÃO:
 *   Hardware : [Ex: PARCODE - Intel Core / PC Pessoal]
 *   Dataset  : [Ex: YearPredictionMSD (N=515345, D=90, K=64, max_iter=50)]
 *   Paralelo CUDA
 *     -  1  thread/bloco: [XX.XXXXXX s]
 *     -  2 threads/bloco: [XX.XXXXXX s]
 *     -  4 threads/bloco: [XX.XXXXXX s]
 *     -  8 threads/bloco: [XX.XXXXXX s]
 *     - 16 threads/bloco: [XX.XXXXXX s]
 *     - 32 threads/bloco: [XX.XXXXXX s]
 *
 * COMPILAÇÃO:
 *   nvcc kmeans_cuda.cu -o kmeans_cuda -O3
 * 
 *
 * EXECUÇÃO:
 *   ./kmeans_cuda <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0> <num_threads_block> [saida.txt]
 *
 * EXEMPLOS:
 *   ./kmeans_cuda/kmeans_cuda Base_de_dados/teste_5k.txt 5001 90 64 50 1 2 centroides_par.txt
 * 
 *   YearPredictionMSD (2 threads por bloco) (descarta coluna 0 = ano):
 *     ./kmeans_cuda/kmeans_cuda Base_de_dados/YearPredictionMSD.txt 515345 90 64 50 1 2 centroides_par.txt
 *
 *   Covertype (2 threads por bloco) (usa todas as 54 colunas numéricas):
 *     ./kmeans_cuda/kmeans_cuda Base_de_dados/covtype.data 581012 54 64 50 0 2 centroides_par.txt
 *
 *   Arquivo pré-convertido em binário 4 threads (extensão .bin):
 *     ./kmeans_cuda/kmeans_cuda Base_de_dados/dados.bin 515345 90 64 50 0 4 centroides_par.txt
 * ============================================================================
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

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
    float *d_pontos;       /*Device N*/
    float *d_centroides;   /*Device centroides*/
    float *d_somas;        /*Device somas*/
    int   *d_cluster_assoc;/*Device cluster association*/
    int   *d_contagens;    /*Device contagens*/

    int   *d_mudancas;     /*Device mudancas: Contagem de mudanças na iteração.*/
} KMeansData;

/* ----------------------------------------------------------------------------
 * Cronometria monotônica de alta precisão (segundos)
 * ---------------------------------------------------------------------------- */
static double tempo_atual_seg(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // <-- Linux
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    //return (omp_get_wtime()); // <-- Windows
}

/* ----------------------------------------------------------------------------
 * Alocação e liberação dos vetores 1D contíguos
 * ---------------------------------------------------------------------------- */

/* Retorna se houve erro (-1: houve erro; 0: não houve erro)*/
static int verificar_cuda(cudaError_t erro, const char *operacao) {
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s: %s\n", operacao, cudaGetErrorString(erro));
        return -1;
    }
    return 0;
}

static int alocar_kmeans(KMeansData *km, int N, int D, int K) {
    km->N = N;
    km->D = D;
    km->K = K;
    km->d_pontos = NULL;
    km->d_centroides = NULL;
    km->d_somas = NULL;
    km->d_cluster_assoc = NULL;
    km->d_contagens = NULL;
    km->d_mudancas = NULL;

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

    // Para cada variável device, alocamos e testamos se houve erro
    // Meio feio mas fica mais compacto escrito assim
    if (verificar_cuda(cudaMalloc((void **)&km->d_pontos, (size_t)N * (size_t)D * sizeof(float)), "Alocar pontos") ||
        verificar_cuda(cudaMalloc((void **)&km->d_centroides, (size_t)K * (size_t)D * sizeof(float)), "Alocar centroides") ||
        verificar_cuda(cudaMalloc((void **)&km->d_somas, (size_t)K * (size_t)D * sizeof(float)), "Alocar somas") ||
        verificar_cuda(cudaMalloc((void **)&km->d_cluster_assoc, (size_t)N * sizeof(int)), "Alocar associações") ||
        verificar_cuda(cudaMalloc((void **)&km->d_contagens, (size_t)K * sizeof(int)), "Alocar contagens") ||
        verificar_cuda(cudaMalloc((void **)&km->d_mudancas, sizeof(int)), "Alocar contador de mudanças")) {
        return -1;
    }
    
    return 0;
}

static void liberar_kmeans(KMeansData *km) {
    // Liberação device
    if (km->d_pontos) cudaFree(km->d_pontos);
    if (km->d_centroides) cudaFree(km->d_centroides);
    if (km->d_somas) cudaFree(km->d_somas);
    if (km->d_cluster_assoc) cudaFree(km->d_cluster_assoc);
    if (km->d_contagens) cudaFree(km->d_contagens);
    if (km->d_mudancas) cudaFree(km->d_mudancas);
    // Liberação host
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

    for (int indice = 0; indice < km->N; indice++) {
        km->cluster_assoc[indice] = -1;
    }
}

// Copia pontos e centróides para device
static int copiar_kmeans_para_device(KMeansData *km) {
    size_t bytes_pontos = (size_t)km->N * (size_t)km->D * sizeof(float);
    size_t bytes_centroides = (size_t)km->K * (size_t)km->D * sizeof(float);
    size_t bytes_associacoes = (size_t)km->N * sizeof(int);

    if (verificar_cuda(cudaMemcpy(km->d_pontos, km->pontos, bytes_pontos, cudaMemcpyHostToDevice), "Copiar pontos para GPU") ||
        verificar_cuda(cudaMemcpy(km->d_centroides, km->centroides, bytes_centroides, cudaMemcpyHostToDevice), "Copiar centroides para GPU") ||
        verificar_cuda(cudaMemcpy(km->d_cluster_assoc, km->cluster_assoc, bytes_associacoes, cudaMemcpyHostToDevice), "Copiar associações para GPU")) {
        return -1;
    }
    return 0;
}

// Copia resultados do device para o host
static int copiar_resultados_para_host(KMeansData *km) {
    size_t bytes_centroides = (size_t)km->K * (size_t)km->D * sizeof(float);
    size_t bytes_associacoes = (size_t)km->N * sizeof(int);
    size_t bytes_contagens = (size_t)km->K * sizeof(int);

    if (verificar_cuda(cudaMemcpy(km->centroides, km->d_centroides, bytes_centroides, cudaMemcpyDeviceToHost), "Copiar centroides para CPU") ||
        verificar_cuda(cudaMemcpy(km->cluster_assoc, km->d_cluster_assoc, bytes_associacoes, cudaMemcpyDeviceToHost), "Copiar associações para CPU") ||
        verificar_cuda(cudaMemcpy(km->contagens, km->d_contagens, bytes_contagens, cudaMemcpyDeviceToHost), "Copiar contagens para CPU")) {
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------------
 * ETAPA 1: ATRIBUIÇÃO (Padrão MAP)
 * Associa cada ponto ao centróide de menor distância euclidiana ao quadrado.
 * Retorna a quantidade de pontos que trocaram de cluster nesta iteração.
 * ---------------------------------------------------------------------------- */

__global__ void etapa_atribuicao_mudancas(int N, int D, int K, const float *pontos, const float *centroides, int *cluster_assoc, int *total_mudancas) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Mudanças no bloco
    extern __shared__ int mudancas_bloco[]; 
    
    int mudou = 0; // Se o ponto associado à thread mudou de centróide

    if (indice < N) { // Garantir acessos dentro da memória
        const float *p = &pontos[(size_t)indice * (size_t)D];
        float menor_dist = FLT_MAX; // Começa no valor máximo para sempre diminuir
        int melhor_c = 0; // Melhor centróide encontrado para o ponto da thread

        for (int c = 0; c < K; c++) {
            const float *cent = &centroides[(size_t)c * (size_t)D];
            float dist_sq = 0.0f;

            /* Distância euclidiana ao quadrado (sem sqrtf: preserva a ordenação) */
            for (int d = 0; d < D; d++) {
                float diff = p[d] - cent[d];
                dist_sq += diff * diff;
            }

            // Se encontrar um centróide melhor
            if (dist_sq < menor_dist) {
                menor_dist = dist_sq;
                melhor_c = c;
            }
        }

        if (cluster_assoc[indice] != melhor_c) {
            cluster_assoc[indice] = melhor_c;
            mudou = 1;
        }
    }

    mudancas_bloco[threadIdx.x] = mudou;

    __syncthreads(); // Espera todas threads do bloco determinarem se teve mudança ou não

    // Soma de todas mudanças no bloco
    for (int passo = 1; passo < blockDim.x; passo *= 2) {
        int indice = 2 * passo * threadIdx.x;
        if (indice + passo < blockDim.x) {
            mudancas_bloco[indice] += mudancas_bloco[indice + passo];
        }
        __syncthreads();
    }

    // Uma thread de cada bloco adiciona valor total do bloco ao somatório global
    if (threadIdx.x == 0 && mudancas_bloco[0] > 0) {
        // Atomic para apenas um bloco interagir com a variável por vez
        atomicAdd(total_mudancas, mudancas_bloco[0]);
    }
}

static int etapa_atribuicao(KMeansData *km, int num_threads) {
    int n_blocks = (km->N + num_threads - 1) / num_threads;
    int mudancas = 0;

    if (verificar_cuda(cudaMemset(km->d_mudancas, 0, sizeof(int)), "Zerar contador de mudanças")) {
        return -1;
    }

    etapa_atribuicao_mudancas<<<n_blocks, num_threads, (size_t)num_threads * sizeof(int)>>>(km->N, km->D, km->K, km->d_pontos, km->d_centroides, km->d_cluster_assoc, km->d_mudancas);

    if (verificar_cuda(cudaGetLastError(), "Lançar kernel de atribuição") ||
        verificar_cuda(cudaMemcpy(&mudancas, km->d_mudancas, sizeof(int), cudaMemcpyDeviceToHost), "Ler contador de mudanças")) {
        return -1;
    }

    return mudancas;
}

/* ----------------------------------------------------------------------------
 * ETAPA 2: ATUALIZAÇÃO DOS CENTRÓIDES (Padrão REDUCE)
 * Soma as coordenadas dos pontos por cluster e recalcula a média aritmética.
 * ---------------------------------------------------------------------------- */

/* Acumulação das coordenadas e contagens */
__global__ void etapa_atualizacao_par_acumulacao(int N, int D, const float *pontos, const int *cluster_assoc, float *somas, int *contagens) {
    size_t indice = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (indice < N) {
        int c = cluster_assoc[indice];
        const float *p = &pontos[(size_t)indice * (size_t)D];

        atomicAdd(&contagens[c], 1);
        for (int d = 0; d < D; d++) {
            atomicAdd(&somas[(size_t)c * (size_t)D + (size_t)d], p[d]);
        }
    }
}

/* Normalização: divide soma pela contagem (mantém centróide se cluster vazio) */
__global__ void etapa_atualizacao_par_normalizacao(int D, int K, const float *somas, float *centroides, const int *contagens) {

    size_t indice = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_coordenadas = (size_t)K * (size_t)D;

    if (indice < total_coordenadas) {
        int c = (int)(indice / (size_t)D);
        if (contagens[c] > 0) {
            centroides[indice] = somas[indice] / (float)contagens[c];
        }
    }
}

static int etapa_atualizacao(KMeansData *km, int num_threads) {
    int N = km->N;
    int K = km->K;

    int n_blocks = (N + num_threads - 1) / num_threads;
    if (verificar_cuda(cudaMemset(km->d_somas, 0, (size_t)K * (size_t)km->D * sizeof(float)), "Zerar somas") ||
        verificar_cuda(cudaMemset(km->d_contagens, 0, (size_t)K * sizeof(int)), "Zerar contagens")) {
        return -1;
    }
    etapa_atualizacao_par_acumulacao<<<n_blocks, num_threads>>>(km->N, km->D, km->d_pontos, km->d_cluster_assoc, km->d_somas, km->d_contagens);

    if (verificar_cuda(cudaGetLastError(), "Lançar kernel de acumulação")) {
        return -1;
    }

    size_t total_coordenadas = (size_t)K * (size_t)km->D;
    n_blocks = (int)((total_coordenadas + (size_t)num_threads - 1) / (size_t)num_threads);

    etapa_atualizacao_par_normalizacao<<<n_blocks, num_threads>>>(km->D, km->K, km->d_somas, km->d_centroides, km->d_contagens);
    
    return verificar_cuda(cudaGetLastError(), "Lançar kernel de normalização");
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
    if (argc < 8) {
        fprintf(stderr,
                "Uso: %s <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0> <num_threads_block> [saida.txt]\n",
                argv[0]);
        return 1;
    }

    const char *arquivo_entrada = argv[1];
    int N         = atoi(argv[2]);
    int D         = atoi(argv[3]);
    int K         = atoi(argv[4]);
    int max_iter  = atoi(argv[5]);
    int skip_col0 = atoi(argv[6]);
    int num_threads_block = atoi(argv[7]);
    const char *arquivo_saida = (argc >= 9) ? argv[8] : "centroides_par_c.txt";

    if (N <= 0 || D <= 0 || K <= 0 || K > N || max_iter <= 0 || num_threads_block <= 0) {
        fprintf(stderr, "Erro: parâmetros inválidos (verifique N, D, K <= N e max_iter).\n");
        return 1;
    }

    KMeansData km = {0};
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
    if (copiar_kmeans_para_device(&km) != 0) {
        liberar_kmeans(&km);
        return 1;
    }

    /* ========================================================================
     * INÍCIO DA REGIÃO CRONOMETRADA (COMPUTAÇÃO K-MEANS)
     * ======================================================================== */
    double t_ini = tempo_atual_seg();

    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        int mudancas = etapa_atribuicao(&km, num_threads_block);
        if (mudancas < 0 || etapa_atualizacao(&km, num_threads_block) != 0) {
            liberar_kmeans(&km);
            return 1;
        }

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
    if (copiar_resultados_para_host(&km) != 0) {
        liberar_kmeans(&km);
        return 1;
    }
    double sse = calcular_sse(&km);
    salvar_centroides(arquivo_saida, &km);

    printf("=== K-means Paralelo (%d threads/bloco) ===\n", num_threads_block);
    printf("Entrada          : %s (N=%d, D=%d, K=%d)\n", arquivo_entrada, N, D, K);
    printf("Iteracoes        : %d / %d\n", iter, max_iter);
    printf("Tempo de I/O     : %.6f s\n", t_io_fim - t_io_ini);
    printf("Tempo K-means    : %.6f s\n", t_fim - t_ini);
    printf("SSE final        : %.4e\n", sse);
    printf("Centroides salvos: %s\n", arquivo_saida);

    liberar_kmeans(&km);
    return 0;
}
#!/usr/bin/env bash
# ==============================================================================
# Projeto 01 - Implementação Paralela do Algoritmo K-means
# Script: executar.sh
# Descrição: Compila e executa a versão sequencial (kmeans_seq.c) e, caso
#            estejam presentes na pasta, roda também as versões OpenMP
#            (1, 2, 4, 8, 16 e 32 threads) e CUDA, validando as saídas.
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 1. Parâmetros padrão (Dataset: YearPredictionMSD)
#    Para Covertype, passe via argumento:
#    ./executar.sh covtype.data 581012 54 64 80 0
# ------------------------------------------------------------------------------
ARQUIVO_DADOS="${1:-YearPredictionMSD.txt}"
N="${2:-515345}"
D="${3:-90}"
K="${4:-64}"
MAX_ITER="${5:-50}"
SKIP_COL0="${6:-1}"

# Seleciona o compilador C disponível (gcc padrão ou gcc-8 no PARCODE)
CC="gcc"
if ! command -v gcc >/dev/null 2>&1 && command -v gcc-8 >/dev/null 2>&1; then
    CC="gcc-8"
fi

CFLAGS="-O3 -std=c99 -Wall -Wextra"

echo "=================================================================="
echo " K-MEANS - EXECUÇÃO E BENCHMARK"
echo "=================================================================="
echo " Base de dados : $ARQUIVO_DADOS"
echo " Parâmetros    : N=$N | D=$D | K=$K | max_iter=$MAX_ITER | skip_col0=$SKIP_COL0"
echo " Compilador    : $CC $CFLAGS"
echo "=================================================================="

# Verifica se a base de dados existe antes de prosseguir
if [ ! -f "$ARQUIVO_DADOS" ]; then
    echo "ERRO: Arquivo '$ARQUIVO_DADOS' não encontrado no diretório atual."
    echo "Uso: $0 <arquivo_entrada> <N> <D> <K> <max_iter> <skip_col0>"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Compilação e Execução — Versão Sequencial (Referência)
# ------------------------------------------------------------------------------
if [ ! -f "kmeans_seq.c" ]; then
    echo "ERRO: 'kmeans_seq.c' não encontrado."
    exit 1
fi

echo ""
echo "[1/3] Compilando kmeans_seq.c..."
$CC $CFLAGS kmeans_seq.c -o kmeans_seq

echo "[1/3] Executando kmeans_seq..."
./kmeans_seq "$ARQUIVO_DADOS" "$N" "$D" "$K" "$MAX_ITER" "$SKIP_COL0" centroides_seq.txt

# ------------------------------------------------------------------------------
# 3. Compilação e Benchmark — Versão OpenMP (se kmeans_omp.c existir)
# ------------------------------------------------------------------------------
if [ -f "kmeans_omp.c" ]; then
    echo ""
    echo "[2/3] Compilando kmeans_omp.c..."
    $CC $CFLAGS -fopenmp kmeans_omp.c -o kmeans_omp

    for THREADS in 1 2 4 8 16 32; do
        echo ""
        echo "--- Executando OpenMP com $THREADS thread(s) ---"
        OMP_NUM_THREADS=$THREADS ./kmeans_omp "$ARQUIVO_DADOS" "$N" "$D" "$K" "$MAX_ITER" "$SKIP_COL0" "centroides_omp_${THREADS}.txt"

        if diff -q centroides_seq.txt "centroides_omp_${THREADS}.txt" >/dev/null; then
            echo "Validação (diff): OK (idêntico ao sequencial)"
        else
            echo "Validação (diff): AVISO - diferença encontrada em centroides_omp_${THREADS}.txt"
        fi
    done
fi

# ------------------------------------------------------------------------------
# 4. Compilação e Execução — Versão CUDA (se kmeans_cuda.cu e nvcc existirem)
# ------------------------------------------------------------------------------
if [ -f "kmeans_cuda.cu" ]; then
    echo ""
    if command -v nvcc >/dev/null 2>&1; then
        echo "[3/3] Compilando kmeans_cuda.cu..."
        nvcc -O3 kmeans_cuda.cu -o kmeans_cuda

        echo "[3/3] Executando kmeans_cuda..."
        ./kmeans_cuda "$ARQUIVO_DADOS" "$N" "$D" "$K" "$MAX_ITER" "$SKIP_COL0" centroides_cuda.txt

        if diff -q centroides_seq.txt centroides_cuda.txt >/dev/null; then
            echo "Validação (diff): OK (idêntico ao sequencial)"
        else
            echo "Validação (diff): AVISO - pequenas diferenças de ponto flutuante (verifique tolerância)"
        fi
    else
        echo "[3/3] AVISO: 'nvcc' não encontrado neste ambiente. Pulando etapa CUDA."
    fi
fi

echo ""
echo "=================================================================="
echo " Execução concluída!"
echo "=================================================================="
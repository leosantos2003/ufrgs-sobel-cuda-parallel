// Filtro de Sobel em CUDA C++ (PGM P5, tons de cinza)

#define _POSIX_C_SOURCE 200809L

// CUDA 1: disponibiliza kernels e memória unificada
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "pgm.h"

static double agora_segundos(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// CUDA 2: __global__ faz esta função executar na GPU.
// Cada thread executa o mesmo cálculo de um pixel que seria feito por uma
// iteração dos dois laços de sobel_sequencial.

__global__ void sobel_cuda(const unsigned char *entrada, unsigned char *saida,
                           int largura, int altura) {
    
    // CUDA 3: substitui os laços y/x por um índice único da grade.
    // Fórmula: blockIdx.x * blockDim.x + threadIdx.x

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = largura * altura;
    if (i >= n) return; // A última grade pode ter threads sem pixel associado.

    int y = i / largura;
    int x = i % largura;
    if (x == 0 || y == 0 || x == largura - 1 || y == altura - 1) {
        saida[i] = 0; // Borda: preta
        return;
    }

    int a = entrada[(y - 1) * largura + (x - 1)];
    int b = entrada[(y - 1) * largura + x];
    int c = entrada[(y - 1) * largura + (x + 1)];
    int d = entrada[y * largura + (x - 1)];
    int f = entrada[y * largura + (x + 1)];
    int g = entrada[(y + 1) * largura + (x - 1)];
    int h = entrada[(y + 1) * largura + x];
    int j = entrada[(y + 1) * largura + (x + 1)];

    // Máscaras Sobel 3x3: gradientes horizontal e vertical
    int gx = -a + c - 2 * d + 2 * f - g + j;
    int gy = -a - 2 * b - c + g + 2 * h + j;
    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    if (magnitude > 255.0f) magnitude = 255.0f;
    saida[i] = (unsigned char)(magnitude + 0.5f);
}



int main(int argc, char **argv) {
    ImagemPGM entrada = {0, 0, NULL};
    ImagemPGM saida = {0, 0, NULL};
    int repeticoes = 10;
    size_t n;
    double inicio, fim;
    
    int ok;
    
    // CUDA 4: preserva o ponteiro host antes de usar memória unificada
    unsigned char *entrada_host;
    size_t threads_por_bloco = 256;
    size_t numero_de_blocos;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s entrada.pgm saida.pgm [repeticoes]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4 && !ler_repeticoes(argv[3], &repeticoes)) return EXIT_FAILURE;
    if (!ler_pgm(argv[1], &entrada)) return EXIT_FAILURE;

    n = (size_t)entrada.largura * (size_t)entrada.altura;
    saida.largura = entrada.largura;
    saida.altura = entrada.altura;

    // CUDA 5: cudaMallocManaged cria memória acessivel pela CPU e
    // pela GPU. A imagem é lida inicialmente por ler_pgm em memória comum;
    // por isso ela é copiada uma única vez para a memória unificada.

    entrada_host = entrada.pixels;
    cudaMallocManaged((void **)&entrada.pixels, n);
    memcpy(entrada.pixels, entrada_host, n);
    free(entrada_host);
    cudaMallocManaged((void **)&saida.pixels, n);

    // CUDA 6: a grade cobre todos os pixels, uma thread por pixel
    numero_de_blocos = (n + threads_por_bloco - 1) / threads_por_bloco;

    // CUDA 7: aquecimento e sincronização antes da região temporizada
    sobel_cuda<<<numero_de_blocos, threads_por_bloco>>>(
        entrada.pixels, saida.pixels, entrada.largura, entrada.altura);
    cudaDeviceSynchronize();

    inicio = agora_segundos();
    for (int i = 0; i < repeticoes; ++i)
        
        // CUDA 8: a chamada <<<...>>> inicia o kernel na GPU
        sobel_cuda<<<numero_de_blocos, threads_por_bloco>>>(
            entrada.pixels, saida.pixels, entrada.largura, entrada.altura);
    
    // CUDA 9: espera o kernel terminar antes de encerrar a medição
    cudaDeviceSynchronize();
    fim = agora_segundos();

    printf("Imagem: %d x %d pixels\n", entrada.largura, entrada.altura);
    printf("Repeticoes: %d\n", repeticoes);
    printf("Tempo medio GPU (kernel + sincronizacao): %.6f ms\n",
           1000.0 * (fim - inicio) / repeticoes);

    ok = escrever_pgm(argv[2], &saida);
    
    // CUDA 10: libera a memória unificada com cudaFree
    cudaFree(saida.pixels);
    cudaFree(entrada.pixels);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
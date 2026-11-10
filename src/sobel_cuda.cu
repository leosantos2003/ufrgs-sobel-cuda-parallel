/* Filtro de Sobel em CUDA C++ (arquivo .cu; I/O compartilhado com a versao C). */
#define _POSIX_C_SOURCE 200809L
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "pgm.h"

static double agora_segundos(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Cada thread CUDA calcula um pixel da imagem de saida. */
__global__ void sobel_kernel(const unsigned char *entrada, unsigned char *saida,
                             int largura, int altura) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= largura || y >= altura) return;

    int i = y * largura + x;
    if (x == 0 || y == 0 || x == largura - 1 || y == altura - 1) {
        saida[i] = 0;
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

    int gx = -a + c - 2 * d + 2 * f - g + j;
    int gy = -a - 2 * b - c + g + 2 * h + j;
    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    if (magnitude > 255.0f) magnitude = 255.0f;
    saida[i] = (unsigned char)(magnitude + 0.5f);
}

static int executar_sobel_cuda(const ImagemPGM *entrada, ImagemPGM *saida,
                              int repeticoes) {
    unsigned char *d_entrada = NULL;
    unsigned char *d_saida = NULL;
    cudaEvent_t inicio_evento = NULL;
    cudaEvent_t fim_evento = NULL;
    size_t n = (size_t)entrada->largura * (size_t)entrada->altura;
    dim3 bloco(16, 16);
    dim3 grade(((unsigned)entrada->largura + bloco.x - 1) / bloco.x,
               ((unsigned)entrada->altura + bloco.y - 1) / bloco.y);
    float kernel_ms = 0.0f;
    double soma_kernel_ms = 0.0, t0, t1;
    int sucesso = 0;
    cudaError_t erro;

#define CUDA_VERIFICAR(chamada) do {                                           \
    erro = (chamada);                                                        \
    if (erro != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA: %s (linha %d): %s\n", #chamada, __LINE__,     \
                cudaGetErrorString(erro));                                    \
        goto limpar;                                                         \
    }                                                                        \
} while (0)

    /* Inicializa CUDA e aloca memoria fora da regiao temporizada. */
    CUDA_VERIFICAR(cudaFree(0));
    CUDA_VERIFICAR(cudaMalloc((void **)&d_entrada, n));
    CUDA_VERIFICAR(cudaMalloc((void **)&d_saida, n));
    CUDA_VERIFICAR(cudaEventCreate(&inicio_evento));
    CUDA_VERIFICAR(cudaEventCreate(&fim_evento));

    /* Aquecimento da GPU fora da medicao (JIT/contexto/cache). */
    CUDA_VERIFICAR(cudaMemcpy(d_entrada, entrada->pixels, n, cudaMemcpyHostToDevice));
    sobel_kernel<<<grade, bloco>>>(d_entrada, d_saida, entrada->largura, entrada->altura);
    CUDA_VERIFICAR(cudaGetLastError());
    CUDA_VERIFICAR(cudaDeviceSynchronize());

    t0 = agora_segundos();
    for (int i = 0; i < repeticoes; ++i) {
        /* Tempo total inclui H2D, kernel, D2H e sincronizacao. */
        CUDA_VERIFICAR(cudaMemcpy(d_entrada, entrada->pixels, n, cudaMemcpyHostToDevice));
        CUDA_VERIFICAR(cudaEventRecord(inicio_evento));
        sobel_kernel<<<grade, bloco>>>(d_entrada, d_saida, entrada->largura, entrada->altura);
        CUDA_VERIFICAR(cudaGetLastError());
        CUDA_VERIFICAR(cudaEventRecord(fim_evento));
        CUDA_VERIFICAR(cudaMemcpy(saida->pixels, d_saida, n, cudaMemcpyDeviceToHost));
        CUDA_VERIFICAR(cudaEventElapsedTime(&kernel_ms, inicio_evento, fim_evento));
        soma_kernel_ms += kernel_ms;
    }
    t1 = agora_segundos();

    printf("Imagem: %d x %d pixels\n", entrada->largura, entrada->altura);
    printf("Repeticoes: %d\n", repeticoes);
    printf("Bloco CUDA: %u x %u threads\n", bloco.x, bloco.y);
    printf("Tempo medio GPU (somente kernel): %.6f ms\n",
           soma_kernel_ms / repeticoes);
    printf("Tempo medio GPU (transferencias + kernel): %.6f ms\n",
           1000.0 * (t1 - t0) / repeticoes);
    sucesso = 1;

limpar:
    if (fim_evento) cudaEventDestroy(fim_evento);
    if (inicio_evento) cudaEventDestroy(inicio_evento);
    if (d_saida) cudaFree(d_saida);
    if (d_entrada) cudaFree(d_entrada);
    return sucesso;
#undef CUDA_VERIFICAR
}

int main(int argc, char **argv) {
    ImagemPGM entrada = {0, 0, NULL};
    ImagemPGM saida = {0, 0, NULL};
    int repeticoes = 10;
    int ok;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s entrada.pgm saida.pgm [repeticoes]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4 && !ler_repeticoes(argv[3], &repeticoes)) return EXIT_FAILURE;
    if (!ler_pgm(argv[1], &entrada)) return EXIT_FAILURE;

    saida.largura = entrada.largura;
    saida.altura = entrada.altura;
    saida.pixels = (unsigned char *)malloc((size_t)saida.largura * saida.altura);
    if (!saida.pixels) {
        fprintf(stderr, "Falha ao alocar memoria para a saida.\n");
        free(entrada.pixels);
        return EXIT_FAILURE;
    }

    ok = executar_sobel_cuda(&entrada, &saida, repeticoes);
    if (ok) ok = escrever_pgm(argv[2], &saida);
    free(saida.pixels);
    free(entrada.pixels);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

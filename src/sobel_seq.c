/* Filtro de Sobel sequencial em C (PGM P5, tons de cinza). */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <time.h>
#include "pgm.h"

static double agora_segundos(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void sobel_sequencial(const unsigned char *entrada, unsigned char *saida,
                            int largura, int altura) {
    for (int y = 0; y < altura; ++y) {
        for (int x = 0; x < largura; ++x) {
            int i = y * largura + x;
            if (x == 0 || y == 0 || x == largura - 1 || y == altura - 1) {
                saida[i] = 0; /* Borda: preta. */
                continue;
            }

            int a = entrada[(y - 1) * largura + (x - 1)];
            int b = entrada[(y - 1) * largura + x];
            int c = entrada[(y - 1) * largura + (x + 1)];
            int d = entrada[y * largura + (x - 1)];
            int f = entrada[y * largura + (x + 1)];
            int g = entrada[(y + 1) * largura + (x - 1)];
            int h = entrada[(y + 1) * largura + x];
            int j = entrada[(y + 1) * largura + (x + 1)];

            /* Mascaras Sobel 3x3: gradientes horizontal e vertical. */
            int gx = -a + c - 2 * d + 2 * f - g + j;
            int gy = -a - 2 * b - c + g + 2 * h + j;
            float magnitude = sqrtf((float)(gx * gx + gy * gy));
            if (magnitude > 255.0f) magnitude = 255.0f;
            saida[i] = (unsigned char)(magnitude + 0.5f);
        }
    }
}

int main(int argc, char **argv) {
    ImagemPGM entrada = {0, 0, NULL};
    ImagemPGM saida = {0, 0, NULL};
    int repeticoes = 10;
    size_t n;
    double inicio, fim;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s entrada.pgm saida.pgm [repeticoes]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4 && !ler_repeticoes(argv[3], &repeticoes)) return EXIT_FAILURE;
    if (!ler_pgm(argv[1], &entrada)) return EXIT_FAILURE;

    n = (size_t)entrada.largura * (size_t)entrada.altura;
    saida.largura = entrada.largura;
    saida.altura = entrada.altura;
    saida.pixels = (unsigned char *)malloc(n);
    if (!saida.pixels) {
        fprintf(stderr, "Falha ao alocar memoria para a saida.\n");
        free(entrada.pixels);
        return EXIT_FAILURE;
    }

    /* Aquecimento fora da regiao temporizada. */
    sobel_sequencial(entrada.pixels, saida.pixels, entrada.largura, entrada.altura);
    inicio = agora_segundos();
    for (int i = 0; i < repeticoes; ++i)
        sobel_sequencial(entrada.pixels, saida.pixels, entrada.largura, entrada.altura);
    fim = agora_segundos();

    printf("Imagem: %d x %d pixels\n", entrada.largura, entrada.altura);
    printf("Repeticoes: %d\n", repeticoes);
    printf("Tempo medio CPU (somente Sobel): %.6f ms\n",
           1000.0 * (fim - inicio) / repeticoes);

    int ok = escrever_pgm(argv[2], &saida);
    free(saida.pixels);
    free(entrada.pixels);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

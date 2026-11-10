#ifndef SOBEL_PGM_H
#define SOBEL_PGM_H

/* Leitura/escrita simples de imagens PGM binarias (P5, 8 bits). */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int largura;
    int altura;
    unsigned char *pixels;
} ImagemPGM;

static int proximo_token(FILE *f, char *token, size_t capacidade) {
    int c;
    size_t i = 0;

    /* Comentarios e espacos em branco sao permitidos no cabecalho. */
    do {
        c = fgetc(f);
        if (c == '#') {
            do { c = fgetc(f); } while (c != '\n' && c != EOF);
        }
    } while (c != EOF && isspace((unsigned char)c));

    if (c == EOF) return 0;

    while (c != EOF && !isspace((unsigned char)c) && c != '#') {
        if (i + 1 >= capacidade) return 0;
        token[i++] = (char)c;
        c = fgetc(f);
    }
    token[i] = '\0';

    if (c == '#') {
        /* Comentario apos um token. */
        do { c = fgetc(f); } while (c != '\n' && c != EOF);
    } else if (c == '\r') {
        /* Aceita cabecalhos Windows com \r\n. */
        int seguinte = fgetc(f);
        if (seguinte != '\n' && seguinte != EOF) ungetc(seguinte, f);
    }
    return i != 0;
}

static int ler_inteiro_positivo(const char *s, int *valor) {
    char *fim;
    long n;
    errno = 0;
    n = strtol(s, &fim, 10);
    if (errno || *fim || n <= 0 || n > INT_MAX) return 0;
    *valor = (int)n;
    return 1;
}

static int ler_pgm(const char *caminho, ImagemPGM *img) {
    FILE *f = fopen(caminho, "rb");
    char token[64];
    int maxvalor;
    size_t n;

    img->largura = img->altura = 0;
    img->pixels = NULL;
    if (!f) {
        perror(caminho);
        return 0;
    }
    if (!proximo_token(f, token, sizeof(token)) || strcmp(token, "P5") != 0 ||
        !proximo_token(f, token, sizeof(token)) ||
        !ler_inteiro_positivo(token, &img->largura) ||
        !proximo_token(f, token, sizeof(token)) ||
        !ler_inteiro_positivo(token, &img->altura) ||
        !proximo_token(f, token, sizeof(token)) ||
        !ler_inteiro_positivo(token, &maxvalor) || maxvalor != 255) {
        fprintf(stderr, "Formato invalido: esperado PGM P5 com maxvalor 255: %s\n", caminho);
        fclose(f);
        return 0;
    }

    if ((size_t)img->altura > SIZE_MAX / (size_t)img->largura) {
        fprintf(stderr, "Dimensoes grandes demais.\n");
        fclose(f);
        return 0;
    }
    n = (size_t)img->largura * (size_t)img->altura;
    img->pixels = (unsigned char *)malloc(n);
    if (!img->pixels) {
        fprintf(stderr, "Falha ao alocar %zu bytes.\n", n);
        fclose(f);
        return 0;
    }
    if (fread(img->pixels, 1, n, f) != n) {
        fprintf(stderr, "Arquivo PGM truncado: %s\n", caminho);
        free(img->pixels);
        img->pixels = NULL;
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int escrever_pgm(const char *caminho, const ImagemPGM *img) {
    FILE *f = fopen(caminho, "wb");
    size_t n = (size_t)img->largura * (size_t)img->altura;
    int ok;
    if (!f) {
        perror(caminho);
        return 0;
    }
    ok = fprintf(f, "P5\n%d %d\n255\n", img->largura, img->altura) > 0 &&
         fwrite(img->pixels, 1, n, f) == n;
    if (fclose(f) != 0) ok = 0;
    if (!ok) fprintf(stderr, "Falha ao gravar: %s\n", caminho);
    return ok;
}

static int ler_repeticoes(const char *s, int *valor) {
    if (!ler_inteiro_positivo(s, valor) || *valor > 1000000) {
        fprintf(stderr, "Repeticoes deve ser um inteiro entre 1 e 1000000.\n");
        return 0;
    }
    return 1;
}

#endif

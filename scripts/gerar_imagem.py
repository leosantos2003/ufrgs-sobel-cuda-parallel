#!/usr/bin/env python3
"""Cria imagem PGM P5 sintetica para testar Sobel; apenas biblioteca padrao."""
import argparse

p = argparse.ArgumentParser()
p.add_argument('largura', type=int)
p.add_argument('altura', type=int)
p.add_argument('arquivo')
a = p.parse_args()
if a.largura <= 0 or a.altura <= 0:
    p.error('A largura e a altura precisam ser positivas')

with open(a.arquivo, 'wb') as f:
    f.write(f'P5\n{a.largura} {a.altura}\n255\n'.encode())
    for y in range(a.altura):
        linha = bytearray(a.largura)
        for x in range(a.largura):
            quad = 120 if ((x // 32 + y // 32) % 2 == 0) else 0
            rampa = (3 * x + 2 * y) % 136
            linha[x] = min(255, quad + rampa)
        f.write(linha)
print(f'Imagem criada: {a.arquivo} ({a.largura}x{a.altura})')

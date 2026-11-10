#!/usr/bin/env python3
"""Compara dois arquivos PGM P5, imprimindo erro absoluto maximo."""
import argparse

p = argparse.ArgumentParser()
p.add_argument('cpu')
p.add_argument('gpu')
p.add_argument('--tolerancia', type=int, default=1)
a = p.parse_args()

def ler_pgm(caminho):
    with open(caminho, 'rb') as f:
        def token():
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Cabecalho incompleto')
                if c == b'#':
                    f.readline()
                    continue
                if c not in b' \t\r\n':
                    break
            b = bytearray()
            while c and c not in b' \t\r\n':
                b.extend(c)
                c = f.read(1)
            return bytes(b)
        tipo = token()
        largura = int(token())
        altura = int(token())
        maxval = int(token())
        if tipo != b'P5' or maxval != 255:
            raise ValueError('Somente PGM P5 8-bit e suportado')
        pixels = f.read()
        if len(pixels) != largura * altura:
            raise ValueError('Tamanho dos pixels invalido')
        return (largura, altura), pixels

(tam1, im1), (tam2, im2) = ler_pgm(a.cpu), ler_pgm(a.gpu)
if tam1 != tam2:
    raise SystemExit(f'Dimensoes diferentes: {tam1} vs {tam2}')
erro_max = max((abs(x - y) for x, y in zip(im1, im2)), default=0)
diferentes = sum(x != y for x, y in zip(im1, im2))
print(f'Pixels diferentes: {diferentes}/{len(im1)}')
print(f'Erro absoluto maximo: {erro_max}')
if erro_max > a.tolerancia:
    raise SystemExit(f'ERRO: diferenca maior que a tolerancia ({a.tolerancia})')
print('OK: saidas concordam dentro da tolerancia.')

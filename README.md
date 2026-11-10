# Sobel 3x3: C sequencial e CUDA

Este projeto implementa a **mesma** detecção de bordas por Sobel em C e CUDA.

- `src/sobel_seq.c`: cálculo sequencial em CPU.
- `src/sobel_cuda.cu`: um pixel de saída por thread CUDA (grade bidimensional, blocos 16x16).
- `src/pgm.h`: leitura e escrita compartilhada do formato PGM binário de 8 bits.
- `pics-in/`: imagens de entrada coloridas.
- `pics-out/`: imagens produzidas pelo Sobel.
- `scripts/gerar_imagem.py`: cria imagens sintéticas de tamanhos diferentes.
- `scripts/comparar.py`: compara as imagens de saída, tolerando diferenças de arredondamento de até 1 nível de cinza.
- `scripts/processar_imagem.py`: aplica o Sobel a PNG/JPG/JPEG/TIFF e gera uma imagem final visualizável.

## O algoritmo: filtro de Sobel 3x3

O **Sobel** é um algoritmo de processamento de imagens que realça **bordas**: regiões nas quais a intensidade dos pixels muda de forma abrupta. Em uma foto em tons de cinza, por exemplo, essas mudanças normalmente correspondem aos contornos de objetos, letras, linhas e limites entre áreas claras e escuras. A imagem de saída não é uma cópia colorida ou suavizada da entrada; ela representa a **força das bordas** encontradas em cada posição.

Para calcular cada pixel de saída, o algoritmo observa uma janela 3x3 centrada no pixel correspondente da imagem de entrada. Nessa janela, ele aplica duas máscaras (convoluções):

```text
Gx = [-1  0  1]       Gy = [-1 -2 -1]
     [-2  0  2]            [ 0  0  0]
     [-1  0  1]            [ 1  2  1]
```

- `Gx` mede a variação da intensidade da esquerda para a direita e, portanto, destaca principalmente bordas verticais.
- `Gy` mede a variação de cima para baixo e destaca principalmente bordas horizontais.

Em seguida, o programa combina os dois gradientes com a magnitude

```text
magnitude = sqrt(Gx² + Gy²)
```

Quanto maior essa magnitude, mais forte é a transição local e mais claro será o pixel na imagem de saída. O resultado é limitado a 255, pois o PGM usado pelo projeto possui pixels de 8 bits (`0` = preto, `255` = branco), e depois é arredondado para o inteiro mais próximo. Os pixels da borda externa recebem zero (preto), pois não possuem todos os oito vizinhos necessários para formar uma janela 3x3.

## Como a implementação executa o Sobel

As duas versões produzem a mesma imagem e aplicam exatamente as máscaras acima:

- Na versão sequencial, dois laços percorrem a imagem e calculam um pixel de saída por vez na CPU.
- Na versão CUDA, a imagem é copiada para a GPU e cada thread calcula um pixel de saída. A grade é bidimensional e os blocos têm 16x16 threads; threads que ficariam fora das dimensões reais da imagem encerram sem acessar memória inválida.

A implementação CUDA é o baseline com leituras diretamente da memória global: cada thread lê a sua vizinhança 3x3. Assim, ela prioriza a clareza e a equivalência com a versão sequencial, deixando o uso de memória compartilhada como uma otimização posterior.

## Requisitos

- GCC (C11) e `make` para a versão sequencial.
- NVIDIA CUDA Toolkit (`nvcc`) e GPU NVIDIA compatível com CUDA para a versão paralela.
- Python 3 (biblioteca padrão) para gerar imagens de teste e comparar saídas.
- Pillow para usar PNG/JPG/JPEG/TIFF com `scripts/processar_imagem.py`.

Instale o Pillow uma única vez, se necessário:

```bash
python3 -m pip install Pillow
```

## Organização

- `src/`: código-fonte C/CUDA e cabeçalho compartilhado.
- `pics-in/`: imagens de entrada (PNG, JPG, JPEG, TIFF ou PGM).
- `pics-out/`: imagens de saída (PNG, JPG ou JPEG).
- `scripts/`: utilitários Python para gerar imagens e comparar resultados.
- Raiz: `Makefile`, `README.md` e os executáveis gerados pela compilação.

## Compilar

```bash
make cpu       # GCC, compila apenas a versão sequencial
make gpu       # nvcc, compila apenas a versão CUDA
# ou: make      # compila as duas
```

Sem `make`:

```bash
gcc -O3 -std=c11 src/sobel_seq.c -o sobel_seq -lm
nvcc -O3 src/sobel_cuda.cu -o sobel_cuda
```

## Executar

```bash
make teste-cpu
make teste-gpu
python3 scripts/comparar.py pics-out/saida_seq.png pics-out/saida_cuda.png
```

O último argumento é o número de repetições; se omitido, vale 10. As imagens geradas são arquivos PGM P5 (grayscale 8 bits, `maxval=255`). A borda externa de um pixel é preenchida com zero. Na aplicação sequencial e CUDA, a magnitude é `sqrt(Gx*Gx + Gy*Gy)`, saturada em 255 e arredondada para o inteiro mais próximo. Pequenas diferenças numéricas de 1 tom de cinza podem surgir por arredondamento em CPU/GPU.

## Usar imagens PNG, JPG, JPEG ou TIFF

Coloque a imagem em `pics-in/` e execute o script abaixo depois de compilar a versão desejada. Ele converte a entrada para um PGM temporário em tons de cinza, executa o Sobel e grava o resultado em `pics-out/`. Os arquivos temporários são removidos automaticamente.

```bash
make cpu
python3 scripts/processar_imagem.py pics-in/foto.jpg
# Resultado: pics-out/foto_sobel.png
```

As imagens TIFF também podem ser usadas diretamente; por exemplo:

```bash
python3 scripts/processar_imagem.py pics-in/foto.tiff
# Resultado: pics-out/foto_sobel.png
```

Para usar a GPU, compile-a e selecione o modo correspondente:

```bash
make gpu
python3 scripts/processar_imagem.py pics-in/foto.png --modo gpu --repeticoes 30
```

Por padrão, o script não substitui uma saída existente. Para escolher o nome do resultado ou permitir a substituição, use:

```bash
python3 scripts/processar_imagem.py pics-in/foto.jpeg \
  --saida bordas.png --sobrescrever
```

## Benchmark com tamanhos diferentes

```bash
python3 scripts/gerar_imagem.py 1024 1024 pics-in/imagem_1024.pgm
python3 scripts/gerar_imagem.py 2048 2048 pics-in/imagem_2048.pgm
python3 scripts/gerar_imagem.py 4096 4096 pics-in/imagem_4096.pgm
./sobel_seq pics-in/imagem_4096.pgm pics-out/cpu_4096.pgm 30
./sobel_cuda pics-in/imagem_4096.pgm pics-out/gpu_4096.pgm 30
python3 scripts/comparar.py pics-out/cpu_4096.pgm pics-out/gpu_4096.pgm
```

A versão sequencial mostra **tempo de processamento** (sem E/S). A versão CUDA mostra dois tempos: (1) **somente kernel** por meio de `cudaEvent` e (2) **transferências H2D + kernel + D2H**, medido no host, com os buffers GPU previamente alocados. Para speedup fim a fim, prefira usar o segundo tempo; explique que a CPU já opera sobre dados na memória principal e que a GPU exige transferências. Ambas as versões fazem aquecimento antes das medições; leitura e escrita de arquivo não entram nos tempos. Execute no mesmo computador e com a mesma imagem e número de repetições.

## Ideias para otimização CUDA

- Comparar blocos 8x8, 16x16 e 32x8 (altere `dim3 bloco(...)`).
- Usar memória compartilhada (`shared memory`) para carregar tiles com halo de 1 pixel.
- Experimentar acesso via memória somente-leitura/texture e comparar tráfego global.
- Medir o impacto de diferentes resoluções e a razão entre tempo de kernel e tempo de transferência.

**Observação:** a implementação CUDA deste projeto é a versão básica global-memory, sem `shared memory`, para servir de baseline antes das otimizações.

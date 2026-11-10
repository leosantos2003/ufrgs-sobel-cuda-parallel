# Sobel 3x3: C sequencial e CUDA

Este projeto implementa a **mesma** detecção de bordas por Sobel em C e CUDA.

## Teste rápido

Execute os comandos abaixo **na raiz do projeto**. Eles usam a primeira imagem disponível, `pics-in/4.1.01.png`, e mostram o tempo médio no terminal. Os resultados serão gravados em `pics-out/4.1.01_cpu.png` e `pics-out/4.1.01_gpu.png`.

```bash
# 1. Compilar a versão CPU
make cpu

# 2. Rodar a versão CPU
python3 scripts/processar_imagem.py pics-in/pic1.png --modo cpu --saida pic1_cpu.png --sobrescrever

# 3. Compilar a versão GPU
make gpu

# 4. Rodar a versão GPU
python3 scripts/processar_imagem.py pics-in/pic1.png --modo gpu --saida pic1_gpu.png --sobrescrever
```

No Windows, execute em um terminal no qual `gcc`, `nvcc`, GNU Make e Python estejam no `PATH`. Se os comandos instalados se chamarem `mingw32-make` e `py`, substitua apenas `make` por `mingw32-make` e `python3` por `py` nos quatro comandos. No macOS, o teste de CPU funciona normalmente; o teste de GPU requer CUDA e, portanto, uma GPU NVIDIA compatível — algo que os Macs atuais não oferecem.

## Requisitos

- Um compilador C compatível com C11 (`gcc`) e GNU Make para a versão CPU.
- NVIDIA CUDA Toolkit (`nvcc`) e GPU NVIDIA compatível com CUDA para a versão GPU.
- Python 3 e Pillow para processar PNG/JPG/JPEG/TIFF.

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
- Na versão CUDA, cada thread calcula um pixel de saída. A imagem é achatada em um vetor: o índice global da thread identifica o pixel, que é convertido de volta nas coordenadas `x` e `y`. A grade tem blocos de 256 threads; threads cujo índice ultrapassa o total de pixels encerram sem acessar memória inválida.

A implementação CUDA usa memória unificada (`cudaMallocManaged`), cada thread lê diretamente a sua vizinhança 3x3 e a CPU sincroniza antes de usar o resultado. Ela prioriza a clareza e a equivalência com a versão sequencial, deixando o uso de memória compartilhada como uma otimização posterior.

## Outros comandos de terminal

### Instalar Pillow

```bash
# Linux/macOS
python3 -m pip install Pillow

# Windows
py -m pip install Pillow
```

### Compilar as duas versões

```bash
make
```

### Processar outra imagem

Troque `minha_foto.png` pelo arquivo desejado. O argumento opcional `--repeticoes` define quantas execuções entram na média (padrão: 10).

```bash
python3 scripts/processar_imagem.py pics-in/minha_foto.png --modo cpu --repeticoes 30
python3 scripts/processar_imagem.py pics-in/minha_foto.png --modo gpu --repeticoes 30
```

Para escolher o nome da saída ou substituir uma existente:

```bash
python3 scripts/processar_imagem.py pics-in/minha_foto.png --modo cpu \
  --saida bordas.png --sobrescrever
```

### Benchmark com imagens PGM

No Windows, use `py` no lugar de `python3` e `src\sobel_seq.exe` / `src\sobel_cuda.exe` no lugar dos executáveis mostrados abaixo.

```bash
python3 scripts/gerar_imagem.py 1024 1024 pics-in/imagem_1024.pgm
python3 scripts/gerar_imagem.py 2048 2048 pics-in/imagem_2048.pgm
python3 scripts/gerar_imagem.py 4096 4096 pics-in/imagem_4096.pgm
./src/sobel_seq pics-in/imagem_4096.pgm pics-out/cpu_4096.pgm 30
./src/sobel_cuda pics-in/imagem_4096.pgm pics-out/gpu_4096.pgm 30
python3 scripts/comparar.py pics-out/cpu_4096.pgm pics-out/gpu_4096.pgm
```

A versão sequencial mostra **tempo de processamento** (sem E/S). A versão CUDA mede, no host, cada lançamento do kernel seguido de sua sincronização e divide o total pelo número de repetições; a alocação, a leitura e a escrita de arquivo ficam fora da medição. Ambas as versões fazem aquecimento antes das medições. Como a versão CUDA usa memória unificada, não há cópias H2D/D2H explícitas no código. Execute no mesmo computador e com a mesma imagem e número de repetições.

## Ideias para otimização CUDA

- Comparar 128, 256 e 512 threads por bloco (altere `threads_por_bloco`).
- Usar memória compartilhada (`shared memory`) para carregar tiles com halo de 1 pixel.
- Experimentar acesso via memória somente-leitura/texture e comparar tráfego global.
- Medir o impacto de diferentes resoluções.

**Observação:** a implementação CUDA deste projeto usa memória unificada e não usa `shared memory`, para servir de baseline antes das otimizações.

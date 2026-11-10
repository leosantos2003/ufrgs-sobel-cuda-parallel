# Sobel 3x3: C sequencial e CUDA

Este projeto implementa detecção de bordas por Sobel em C e CUDA.

## Teste rápido

Execute na raiz do projeto. Este teste usa `images/pgm-input/pic1.pgm`, portanto **não requer Python nem Pillow**. O tempo médio aparece no terminal; as saídas ficam em `images/pgm-output/`.

```bash
# 1. Compilar e executar a versão CPU
make cpu
./src/sobel_seq images/pgm-input/pic1.pgm images/pgm-output/pic1_cpu.pgm 10

# 2. Compilar e executar a versão GPU
make gpu
./src/sobel_cuda images/pgm-input/pic1.pgm images/pgm-output/pic1_gpu.pgm 10
```

No Windows, use um terminal com `gcc`, `nvcc` e GNU Make no `PATH`. Se necessário, substitua `make` por `mingw32-make`; para executar os binários, use `src\sobel_seq.exe` e `src\sobel_cuda.exe`. No macOS, somente o teste CPU é suportado: CUDA exige uma GPU NVIDIA compatível, indisponível nos Macs atuais.

## Requisitos

- Um compilador C compatível com C11 (`gcc`) e GNU Make para a versão CPU.
- NVIDIA CUDA Toolkit (`nvcc`) e GPU NVIDIA compatível com CUDA para a versão GPU.
- Python 3 e Pillow somente para os testes que usam PNG/JPG/JPEG/TIFF.

## Organização de arquivos e pastas

```text
.
├── images/
│   ├── png-input/       # imagens PNG originais
│   ├── png-output/      # resultados PNG visualizáveis
│   ├── pgm-input/       # entradas PGM para os executáveis
│   └── pgm-output/      # resultados PGM dos testes e benchmarks
├── notebooks/
│   └── sobel_cuda_colab.ipynb  # execução opcional no Google Colab
├── scripts/
│   ├── processar_imagem.py     # PNG/JPG/TIFF -> Sobel -> PNG/JPG
│   ├── gerar_imagem.py         # cria imagens PGM sintéticas
│   └── comparar.py              # compara duas imagens PGM
├── src/
│   ├── sobel_seq.c             # implementação sequencial em C
│   ├── sobel_cuda.cu           # implementação paralela em CUDA
│   ├── pgm.h                   # leitura e escrita de PGM P5
│   └── sobel_seq / sobel_cuda  # executáveis gerados pela compilação
├── cuda_content_learned/       # exemplos e material CUDA da disciplina
├── graphics-material/          # material complementar da disciplina
├── Makefile
└── README.md
```

## Teste rápido com PNG (opcional)

Use esta alternativa para gerar uma saída PNG que pode ser aberta diretamente. Ela requer Pillow; instale-o dentro de uma venv para não alterar o Python global.

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip Pillow

make cpu
python scripts/processar_imagem.py images/png-input/pic1.png --modo cpu --saida pic1_cpu.png --sobrescrever

make gpu
python scripts/processar_imagem.py images/png-input/pic1.png --modo gpu --saida pic1_gpu.png --sobrescrever
```

No Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip Pillow

mingw32-make cpu
python scripts/processar_imagem.py images/png-input/pic1.png --modo cpu --saida pic1_cpu.png --sobrescrever

mingw32-make gpu
python scripts/processar_imagem.py images/png-input/pic1.png --modo gpu --saida pic1_gpu.png --sobrescrever
```

As saídas são gravadas em `images/png-output/`.

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
python3 scripts/processar_imagem.py images/png-input/minha_foto.png --modo cpu --repeticoes 30
python3 scripts/processar_imagem.py images/png-input/minha_foto.png --modo gpu --repeticoes 30
```

Para escolher o nome da saída ou substituir uma existente:

```bash
python3 scripts/processar_imagem.py images/png-input/minha_foto.png --modo cpu \
  --saida bordas.png --sobrescrever
```

### Benchmark com imagens PGM

No Windows, use `py` no lugar de `python3` e `src\sobel_seq.exe` / `src\sobel_cuda.exe` no lugar dos executáveis mostrados abaixo.

```bash
python3 scripts/gerar_imagem.py 1024 1024 images/pgm-input/imagem_1024.pgm
python3 scripts/gerar_imagem.py 2048 2048 images/pgm-input/imagem_2048.pgm
python3 scripts/gerar_imagem.py 4096 4096 images/pgm-input/imagem_4096.pgm
./src/sobel_seq images/pgm-input/imagem_4096.pgm images/pgm-output/cpu_4096.pgm 30
./src/sobel_cuda images/pgm-input/imagem_4096.pgm images/pgm-output/gpu_4096.pgm 30
python3 scripts/comparar.py images/pgm-output/cpu_4096.pgm images/pgm-output/gpu_4096.pgm
```

A versão sequencial mostra **tempo de processamento** (sem E/S). A versão CUDA mede, no host, cada lançamento do kernel seguido de sua sincronização e divide o total pelo número de repetições; a alocação, a leitura e a escrita de arquivo ficam fora da medição. Ambas as versões fazem aquecimento antes das medições. Como a versão CUDA usa memória unificada, não há cópias H2D/D2H explícitas no código. Execute no mesmo computador e com a mesma imagem e número de repetições.

## Ideias para otimização CUDA

- Comparar 128, 256 e 512 threads por bloco (altere `threads_por_bloco`).
- Usar memória compartilhada (`shared memory`) para carregar tiles com halo de 1 pixel.
- Experimentar acesso via memória somente-leitura/texture e comparar tráfego global.
- Medir o impacto de diferentes resoluções.

**Observação:** a implementação CUDA deste projeto usa memória unificada e não usa `shared memory`, para servir de baseline antes das otimizações.

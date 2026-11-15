# Proposta de experimento: Sobel sequencial versus CUDA

## 1. Objetivo

Avaliar como o tamanho de uma imagem influencia o desempenho do filtro de
Sobel implementado de forma sequencial na CPU e em CUDA na GPU. O experimento
usa as implementações já existentes no repositório; não será feita nenhuma
alteração nos arquivos `src/sobel_seq.c` ou `src/sobel_cuda.cu`.

## 2. Questão experimental

Em qual tamanho de imagem a implementação CUDA passa a apresentar menor tempo
médio de processamento que a implementação sequencial?

## 3. Hipótese

Para imagens pequenas, o custo de lançar e sincronizar o kernel pode reduzir a
vantagem da GPU. À medida que o número de pixels aumenta, o paralelismo de uma
thread por pixel deve tornar a versão CUDA relativamente mais rápida que a
versão sequencial.

## 4. Implementações avaliadas

| Identificação | Implementação | Configuração mantida |
|---|---|---|
| CPU | `src/sobel_seq` | Dois laços sequenciais, um pixel por iteração. |
| GPU | `src/sobel_cuda` | Uma thread CUDA por pixel, 256 threads por bloco e memória unificada. |

O número de threads por bloco permanece em 256, pois ele já está definido no
código CUDA. Portanto, o único fator experimental variado será a resolução da
imagem, sem introduzir memória compartilhada ou outros conceitos não cobertos
no conteúdo estudado.

## 5. Variáveis do experimento

- **Variável independente:** resolução da imagem de entrada.
- **Variável dependente:** tempo médio informado pelo executável, em
  milissegundos.
- **Medida derivada:** speedup, calculado por `tempo CPU / tempo GPU`.
- **Variáveis controladas:** mesma máquina, mesma imagem de entrada para as
  duas versões de cada resolução, mesmo número de repetições, mesma versão dos
  executáveis e ausência de outros programas pesados durante a medição.

## 6. Conjunto de entradas

Devem ser geradas imagens PGM sintéticas quadradas, com os seguintes tamanhos:

| Resolução | Número de pixels |
|---:|---:|
| 256 x 256 | 65.536 |
| 512 x 512 | 262.144 |
| 1024 x 1024 | 1.048.576 |
| 2048 x 2048 | 4.194.304 |
| 4096 x 4096 | 16.777.216 |

Caso a GPU não tenha memória suficiente para a maior resolução, o tamanho que
falhar deve ser registrado e omitido das demais medições; os tamanhos menores
continuam válidos.

## 7. Procedimento

1. Compile os dois executáveis uma única vez:

   ```bash
   make
   ```

2. Para cada resolução, gere a imagem de entrada. Exemplo para 2048 x 2048:

   ```bash
   python3 scripts/gerar_imagem.py 2048 2048 images/pgm-input/imagem_2048.pgm
   ```

3. Execute cada versão com 30 repetições internas:

   ```bash
   ./src/sobel_seq images/pgm-input/imagem_2048.pgm images/pgm-output/cpu_2048.pgm 30
   ./src/sobel_cuda images/pgm-input/imagem_2048.pgm images/pgm-output/gpu_2048.pgm 30
   ```

4. Repita o par de comandos acima cinco vezes para cada resolução. Registre o
   tempo médio que cada executável imprime em cada tentativa.

5. Verifique a corretude de cada par de saídas:

   ```bash
   python3 scripts/comparar.py images/pgm-output/cpu_2048.pgm images/pgm-output/gpu_2048.pgm
   ```

6. Calcule, para cada resolução, a média dos cinco tempos obtidos para CPU e
   GPU e o speedup correspondente.

Os programas já executam uma rodada de aquecimento antes da região cronometrada.
A leitura e a escrita dos arquivos, assim como as alocações, ficam fora da
medição. O resultado GPU deve ser descrito exatamente como fornecido pelo
programa: **tempo do kernel mais sincronização por repetição**; ele não é uma
medição exclusiva do kernel por eventos CUDA.

## 8. Tabela para registrar resultados

| Resolução | Tempo médio CPU (ms) | Tempo médio GPU (ms) | Speedup | Saídas idênticas? |
|---:|---:|---:|---:|---|
| 256 x 256 |  |  |  |  |
| 512 x 512 |  |  |  |  |
| 1024 x 1024 |  |  |  |  |
| 2048 x 2048 |  |  |  |  |
| 4096 x 4096 |  |  |  |  |

O speedup é calculado por:

```text
speedup = tempo médio CPU / tempo médio GPU
```

Valores maiores que 1 indicam vantagem da GPU; valores menores que 1 indicam
que a CPU foi mais rápida naquele tamanho de entrada.

## 9. Apresentação e análise esperada

O relatório deve conter:

- a tabela de resultados preenchida;
- um gráfico de resolução versus tempo médio, com uma série para CPU e outra
  para GPU;
- um gráfico de resolução versus speedup;
- a configuração da máquina usada nos testes: modelo da CPU, modelo da GPU,
  memória RAM, versão do CUDA Toolkit, sistema operacional e compiladores;
- a confirmação de que as saídas CPU e GPU são idênticas;
- uma discussão sobre a resolução a partir da qual a GPU foi vantajosa e sobre
  o efeito do overhead em imagens pequenas.

## 10. Escopo em relação ao conteúdo estudado

O experimento usa somente os conceitos já presentes no projeto e no material
da disciplina: kernel CUDA, grid, blocos, threads, cálculo de índice global,
memória unificada e `cudaDeviceSynchronize()`. Memória compartilhada,
`__shared__`, `__syncthreads()` e variações no kernel não fazem parte deste
experimento.

NVCC     ?= nvcc
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -pedantic
# A RTX 4050 deste projeto e Ada (compute capability 8.9). Altere para a
# arquitetura da sua GPU, por exemplo: make -B gpu CUDA_ARCH=sm_86
CUDA_ARCH ?= sm_89
NVFLAGS   ?= -O3 -arch=$(CUDA_ARCH)

ifeq ($(OS),Windows_NT)
EXE := .exe
PYTHON ?= py
# O GNU Make define CC=cc internamente. '?=' nao o substitui, portanto
# escolhemos explicitamente o GCC UCRT64 padrao do MSYS2 quando ele existe.
# Um CC passado pelo usuario (por exemplo, 'make CC=gcc') continua prevalecendo.
MSYS2_ROOT ?= C:/msys64
MSYS2_UCRT_BIN ?= /c/msys64/ucrt64/bin
ifeq ($(origin CC),default)
ifneq ($(wildcard $(MSYS2_ROOT)/ucrt64/bin/gcc.exe),)
CC := $(MSYS2_ROOT)/ucrt64/bin/gcc.exe
else
CC := gcc
endif
endif
# gcc.exe chama cc1.exe e outros auxiliares que dependem das DLLs UCRT64.
# Este make usa o shell do MSYS2, cujo PATH e separado por ':' e usa /c/.
# O ajuste vale somente para os comandos filhos do make; nao altera o terminal.
export PATH := $(MSYS2_UCRT_BIN):$(PATH)
else
PYTHON ?= python3
ifeq ($(origin CC),default)
CC := gcc
endif
endif

EXEC_CPU := src/sobel_seq$(EXE)
EXEC_GPU := src/sobel_cuda$(EXE)

.PHONY: all cpu gpu experimento clean teste-cpu teste-gpu

all: $(EXEC_CPU) $(EXEC_GPU)
cpu: $(EXEC_CPU)
gpu: $(EXEC_GPU)

experimento: all
	$(PYTHON) scripts/executar_experimento.py --sem-compilar

$(EXEC_CPU): src/sobel_seq.c src/pgm.h
	$(CC) $(CFLAGS) src/sobel_seq.c -o $@ -lm

$(EXEC_GPU): src/sobel_cuda.cu src/pgm.h
	$(NVCC) $(NVFLAGS) src/sobel_cuda.cu -o $@

teste-cpu: $(EXEC_CPU)
	$(PYTHON) scripts/processar_imagem.py images/png-input/pic1.png --saida teste_cpu.png --modo cpu --repeticoes 10 --sobrescrever

teste-gpu: $(EXEC_GPU)
	$(PYTHON) scripts/processar_imagem.py images/png-input/pic1.png --saida teste_gpu.png --modo gpu --repeticoes 10 --sobrescrever

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) sobel_seq sobel_cuda sobel_seq.exe sobel_cuda.exe images/png-output/*.png images/png-output/*.jpg images/png-output/*.jpeg images/pgm-output/*.pgm

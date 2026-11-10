CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -pedantic
NVFLAGS  ?= -O3

ifeq ($(OS),Windows_NT)
EXE := .exe
PYTHON ?= py
else
PYTHON ?= python3
endif

EXEC_CPU := src/sobel_seq$(EXE)
EXEC_GPU := src/sobel_cuda$(EXE)

.PHONY: all cpu gpu clean teste-cpu teste-gpu

all: $(EXEC_CPU) $(EXEC_GPU)
cpu: $(EXEC_CPU)
gpu: $(EXEC_GPU)

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

CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -pedantic
NVFLAGS  ?= -O3

ifeq ($(OS),Windows_NT)
EXE := .exe
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
	python3 scripts/processar_imagem.py pics-in/house.png --saida saida_seq.png --modo cpu --repeticoes 10

teste-gpu: $(EXEC_GPU)
	python3 scripts/processar_imagem.py pics-in/house.png --saida saida_cuda.png --modo gpu --repeticoes 10

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) sobel_seq sobel_cuda sobel_seq.exe sobel_cuda.exe pics-out/*.png pics-out/*.jpg pics-out/*.jpeg pics-out/*.pgm

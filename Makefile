CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -pedantic
NVFLAGS  ?= -O3

.PHONY: all cpu gpu clean teste-cpu teste-gpu

all: sobel_seq sobel_cuda
cpu: sobel_seq
gpu: sobel_cuda

sobel_seq: src/sobel_seq.c src/pgm.h
	$(CC) $(CFLAGS) src/sobel_seq.c -o sobel_seq -lm

sobel_cuda: src/sobel_cuda.cu src/pgm.h
	$(NVCC) $(NVFLAGS) src/sobel_cuda.cu -o sobel_cuda

teste-cpu: sobel_seq
	python3 scripts/processar_imagem.py pics-in/house.png --saida saida_seq.png --modo cpu --repeticoes 10

teste-gpu: sobel_cuda
	python3 scripts/processar_imagem.py pics-in/house.png --saida saida_cuda.png --modo gpu --repeticoes 10

clean:
	rm -f sobel_seq sobel_cuda pics-out/*.png pics-out/*.jpg pics-out/*.jpeg pics-out/*.pgm

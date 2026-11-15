#!/usr/bin/env python3
"""Executa o experimento de desempenho CPU x CUDA proposto no projeto.

O script usa somente a biblioteca padrao do Python. Ele gera as entradas,
executa cada programa cinco vezes, valida as saidas, registra as medicoes
brutas e produz um relatorio Markdown e dois graficos SVG acessiveis.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import html
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
TAMANHOS_PADRAO = (256, 512, 1024, 2048, 4096)
AZUL = "#0072B2"       # Paleta Okabe-Ito: distinguivel sem depender de vermelho/verde.
LARANJA = "#D55E00"
VERDE = "#009E73"
PADRAO_TEMPO = re.compile(
    r"Tempo\s+medio\s+(?:CPU|GPU).*?:\s*([0-9]+(?:[\.,][0-9]+)?)\s*ms",
    re.IGNORECASE,
)


class ErroExperimento(RuntimeError):
    """Erro que deve ser mostrado ao usuario sem um traceback ruidoso."""


@dataclass
class ResultadoResolucao:
    tamanho: int
    tempos_cpu: List[float] = field(default_factory=list)
    tempos_gpu: List[float] = field(default_factory=list)
    saidas_identicas: Optional[bool] = None
    erro_gpu: str = ""
    erro_validacao: str = ""

    @property
    def pixels(self) -> int:
        return self.tamanho * self.tamanho

    @property
    def media_cpu(self) -> Optional[float]:
        return statistics.fmean(self.tempos_cpu) if self.tempos_cpu else None

    @property
    def desvio_cpu(self) -> Optional[float]:
        return statistics.stdev(self.tempos_cpu) if len(self.tempos_cpu) > 1 else None

    @property
    def media_gpu(self) -> Optional[float]:
        return statistics.fmean(self.tempos_gpu) if self.tempos_gpu else None

    @property
    def desvio_gpu(self) -> Optional[float]:
        return statistics.stdev(self.tempos_gpu) if len(self.tempos_gpu) > 1 else None

    @property
    def speedup(self) -> Optional[float]:
        if self.media_cpu is None or self.media_gpu in (None, 0):
            return None
        return self.media_cpu / self.media_gpu

    @property
    def tem_medida_gpu_valida(self) -> bool:
        return bool(self.tempos_gpu) and not self.erro_gpu and self.saidas_identicas is True


def argumento_positivo(valor: str) -> int:
    try:
        inteiro = int(valor)
    except ValueError as erro:
        raise argparse.ArgumentTypeError("informe um inteiro positivo") from erro
    if inteiro <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser positivo")
    return inteiro


def criar_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o experimento Sobel sequencial versus CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tamanhos",
        type=argumento_positivo,
        nargs="+",
        default=list(TAMANHOS_PADRAO),
        help="lados das imagens quadradas, em pixels",
    )
    parser.add_argument(
        "--repeticoes",
        type=argumento_positivo,
        default=30,
        help="repeticoes internas solicitadas a cada executavel",
    )
    parser.add_argument(
        "--tentativas",
        type=argumento_positivo,
        default=5,
        help="execucoes independentes de cada implementacao por resolucao",
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RAIZ_PROJETO / "resultados" / "experimento",
        help="pasta para dados, logs, imagens PGM, relatorio e graficos",
    )
    parser.add_argument(
        "--executavel-cpu",
        type=Path,
        help="caminho do executavel sequencial; detectado automaticamente se omitido",
    )
    parser.add_argument(
        "--executavel-gpu",
        type=Path,
        help="caminho do executavel CUDA; detectado automaticamente se omitido",
    )
    parser.add_argument(
        "--sem-compilar",
        action="store_true",
        help="nao executa 'make'; use quando os dois executaveis ja estiverem compilados",
    )
    parser.add_argument(
        "--continuar-apos-falha-gpu",
        action="store_true",
        help="tenta resolucoes maiores mesmo depois de uma falha da GPU",
    )
    return parser


def caminho_absoluto(caminho: Path) -> Path:
    return caminho if caminho.is_absolute() else (RAIZ_PROJETO / caminho)


def executar(
    comando: Sequence[str], *, cwd: Path, descricao: str, log_destino: Optional[Path] = None
) -> str:
    try:
        processo = subprocess.run(
            list(comando),
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as erro:
        raise ErroExperimento(
            f"Nao foi possivel iniciar {descricao}: '{comando[0]}' nao foi encontrado."
        ) from erro
    except OSError as erro:
        if getattr(erro, "winerror", None) == 4551:
            mensagem = (
                f"{descricao} foi bloqueado pela politica de Controle de Aplicativo do Windows. "
                f"O executavel nao foi iniciado: {comando[0]}. Solicite a TI/administrador uma regra "
                "que permita este executavel; nao desative a politica de seguranca."
            )
        else:
            mensagem = f"Nao foi possivel iniciar {descricao}: {erro}"
        if log_destino is not None:
            log_destino.parent.mkdir(parents=True, exist_ok=True)
            log_destino.write_text(mensagem + "\n", encoding="utf-8")
        raise ErroExperimento(mensagem) from erro

    if log_destino is not None:
        log_destino.parent.mkdir(parents=True, exist_ok=True)
        log_destino.write_text(processo.stdout, encoding="utf-8")

    if processo.returncode != 0:
        detalhe = processo.stdout.strip() or "sem saida no terminal"
        raise ErroExperimento(
            f"{descricao} terminou com codigo {processo.returncode}.\n{detalhe}"
        )
    return processo.stdout


def compilar() -> None:
    executar(["make"], cwd=RAIZ_PROJETO, descricao="a compilacao com make")


def localizar_executavel(informado: Optional[Path], nome: str) -> Path:
    if informado is not None:
        candidato = caminho_absoluto(informado)
        if not candidato.is_file():
            raise ErroExperimento(f"Executavel {nome} nao encontrado: {candidato}")
        return candidato

    extensoes = (".exe", "") if os.name == "nt" else ("", ".exe")
    for extensao in extensoes:
        candidato = RAIZ_PROJETO / "src" / f"sobel_{nome}{extensao}"
        if candidato.is_file():
            return candidato
    raise ErroExperimento(
        f"Executavel sobel_{nome} nao encontrado. Execute 'make' ou informe "
        f"--executavel-{nome}."
    )


def gerar_pgm(caminho: Path, tamanho: int) -> None:
    """Gera a mesma imagem sintetica deterministica de gerar_imagem.py."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("wb") as arquivo:
        arquivo.write(f"P5\n{tamanho} {tamanho}\n255\n".encode("ascii"))
        for y in range(tamanho):
            linha = bytearray(tamanho)
            for x in range(tamanho):
                quadriculado = 120 if ((x // 32 + y // 32) % 2 == 0) else 0
                rampa = (3 * x + 2 * y) % 136
                linha[x] = min(255, quadriculado + rampa)
            arquivo.write(linha)


def extrair_tempo(saida: str, implementacao: str) -> float:
    encontrado = PADRAO_TEMPO.search(saida)
    if not encontrado:
        raise ErroExperimento(
            f"Nao encontrei o tempo medio de {implementacao} na saida:\n{saida.strip()}"
        )
    return float(encontrado.group(1).replace(",", "."))


def nome_log(implementacao: str, tamanho: int, tentativa: int) -> str:
    return f"{implementacao}_{tamanho}x{tamanho}_tentativa_{tentativa}.log"


def rodar_implementacao(
    executavel: Path,
    implementacao: str,
    entrada: Path,
    saida: Path,
    repeticoes: int,
    tentativa: int,
    logs: Path,
) -> Tuple[float, str]:
    comando = [str(executavel), str(entrada), str(saida), str(repeticoes)]
    logs.mkdir(parents=True, exist_ok=True)
    texto = executar(
        comando,
        cwd=RAIZ_PROJETO,
        descricao=f"{implementacao} para {entrada.stem}, tentativa {tentativa}",
        log_destino=logs / nome_log(implementacao, int(entrada.stem.split("_")[-1]), tentativa),
    )
    return extrair_tempo(texto, implementacao), texto


def escrever_linha_csv(
    escritor: csv.DictWriter,
    arquivo,
    *,
    tamanho: int,
    implementacao: str,
    tentativa: int,
    tempo_ms: Optional[float],
    status: str,
    detalhe: str = "",
) -> None:
    escritor.writerow(
        {
            "resolucao": f"{tamanho}x{tamanho}",
            "pixels": tamanho * tamanho,
            "implementacao": implementacao,
            "tentativa": tentativa,
            "tempo_medio_ms": "" if tempo_ms is None else f"{tempo_ms:.6f}",
            "status": status,
            "detalhe": detalhe.replace("\n", " "),
        }
    )
    arquivo.flush()


def verificar_saidas(cpu: Path, gpu: Path) -> Tuple[bool, str]:
    comando = [sys.executable, str(RAIZ_PROJETO / "scripts" / "comparar.py"), str(cpu), str(gpu), "--tolerancia", "0"]
    processo = subprocess.run(
        comando,
        cwd=RAIZ_PROJETO,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return processo.returncode == 0, processo.stdout


def executar_medicoes(
    tamanhos: Sequence[int],
    executavel_cpu: Path,
    executavel_gpu: Path,
    repeticoes: int,
    tentativas: int,
    pasta_resultados: Path,
    continuar_apos_falha_gpu: bool,
) -> List[ResultadoResolucao]:
    entradas = pasta_resultados / "entradas"
    saidas = pasta_resultados / "saidas"
    logs = pasta_resultados / "logs"
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    saidas.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    resultados: List[ResultadoResolucao] = []
    falha_de_gpu_anterior = False
    campos_csv = [
        "resolucao",
        "pixels",
        "implementacao",
        "tentativa",
        "tempo_medio_ms",
        "status",
        "detalhe",
    ]

    with (pasta_resultados / "medicoes_brutas.csv").open("w", newline="", encoding="utf-8") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=campos_csv)
        escritor.writeheader()
        arquivo_csv.flush()

        for tamanho in tamanhos:
            resultado = ResultadoResolucao(tamanho=tamanho)
            resultados.append(resultado)
            if falha_de_gpu_anterior and not continuar_apos_falha_gpu:
                resultado.erro_gpu = "Nao executada apos falha de GPU em resolucao menor."
                escrever_linha_csv(
                    escritor, arquivo_csv, tamanho=tamanho, implementacao="GPU", tentativa=0,
                    tempo_ms=None, status="pulada", detalhe=resultado.erro_gpu,
                )
                continue

            entrada = entradas / f"imagem_{tamanho}.pgm"
            saida_cpu = saidas / f"cpu_{tamanho}.pgm"
            saida_gpu = saidas / f"gpu_{tamanho}.pgm"
            gerar_pgm(entrada, tamanho)

            for tentativa in range(1, tentativas + 1):
                try:
                    tempo, _ = rodar_implementacao(
                        executavel_cpu, "CPU", entrada, saida_cpu, repeticoes, tentativa, logs
                    )
                except ErroExperimento as erro:
                    escrever_linha_csv(
                        escritor, arquivo_csv, tamanho=tamanho, implementacao="CPU", tentativa=tentativa,
                        tempo_ms=None, status="falhou", detalhe=str(erro),
                    )
                    raise
                resultado.tempos_cpu.append(tempo)
                escrever_linha_csv(
                    escritor, arquivo_csv, tamanho=tamanho, implementacao="CPU", tentativa=tentativa,
                    tempo_ms=tempo, status="ok",
                )

            for tentativa in range(1, tentativas + 1):
                try:
                    tempo, _ = rodar_implementacao(
                        executavel_gpu, "GPU", entrada, saida_gpu, repeticoes, tentativa, logs
                    )
                except ErroExperimento as erro:
                    resultado.erro_gpu = str(erro)
                    falha_de_gpu_anterior = True
                    escrever_linha_csv(
                        escritor, arquivo_csv, tamanho=tamanho, implementacao="GPU", tentativa=tentativa,
                        tempo_ms=None, status="falhou", detalhe=resultado.erro_gpu,
                    )
                    break
                resultado.tempos_gpu.append(tempo)
                escrever_linha_csv(
                    escritor, arquivo_csv, tamanho=tamanho, implementacao="GPU", tentativa=tentativa,
                    tempo_ms=tempo, status="ok",
                )

            if resultado.tempos_gpu and not resultado.erro_gpu:
                resultado.saidas_identicas, comparacao = verificar_saidas(saida_cpu, saida_gpu)
                (logs / f"comparacao_{tamanho}x{tamanho}.log").write_text(comparacao, encoding="utf-8")
                if resultado.saidas_identicas is False:
                    resultado.erro_validacao = (
                        "Saidas CPU/GPU divergentes; tempos da GPU e speedup foram invalidados."
                    )

    return resultados


def saida_de_comando(comando: Sequence[str]) -> str:
    try:
        processo = subprocess.run(
            list(comando), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
    except FileNotFoundError:
        return "Nao encontrado"
    texto = processo.stdout.strip()
    return texto if texto else f"Codigo de saida {processo.returncode}"


def memoria_ram_total() -> str:
    if os.name == "nt":
        class EstadoMemoria(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        estado = EstadoMemoria()
        estado.dwLength = ctypes.sizeof(EstadoMemoria)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado)):
            return f"{estado.ullTotalPhys / (1024 ** 3):.1f} GiB"
    try:
        return f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3):.1f} GiB"
    except (AttributeError, OSError, ValueError):
        return "Nao identificado"


def coletar_ambiente() -> Dict[str, str]:
    processador = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Nao identificado")
    return {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sistema_operacional": platform.platform(),
        "cpu": processador,
        "ram": memoria_ram_total(),
        "gpu": saida_de_comando([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ]),
        "cuda_toolkit": saida_de_comando(["nvcc", "--version"]),
        "compilador_c": saida_de_comando(["gcc", "--version"]).split("\n")[0],
        "python": sys.version.split("\n")[0],
        "plataforma_python": platform.python_implementation(),
    }


def escapar(texto: object) -> str:
    return html.escape(str(texto), quote=True)


def limite_superior_bonito(valor: float) -> float:
    if valor <= 0:
        return 1.0
    expoente = math.floor(math.log10(valor))
    fracao = valor / (10 ** expoente)
    for passo in (1, 2, 2.5, 5, 10):
        if fracao <= passo:
            return passo * (10 ** expoente)
    return 10 ** (expoente + 1)


def formatar_numero(valor: float) -> str:
    if valor >= 100:
        return f"{valor:.0f}"
    if valor >= 10:
        return f"{valor:.1f}"
    if valor >= 1:
        return f"{valor:.2f}"
    return f"{valor:.3f}"


def cabecalho_svg(titulo: str, descricao: str, largura: int = 1200, altura: int = 720) -> List[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}" role="img" aria-labelledby="titulo descricao">',
        f"  <title id=\"titulo\">{escapar(titulo)}</title>",
        f"  <desc id=\"descricao\">{escapar(descricao)}</desc>",
        "  <style>text { font-family: Arial, Helvetica, sans-serif; fill: #1F2937; } .grade { stroke: #D1D5DB; stroke-width: 1; } .eixo { stroke: #374151; stroke-width: 1.5; } .rotulo { font-size: 18px; } .pequeno { font-size: 15px; fill: #4B5563; } .titulo { font-size: 28px; font-weight: 700; } .subtitulo { font-size: 17px; fill: #4B5563; }</style>",
        '  <rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]


def finalizar_svg(linhas: List[str], destino: Path) -> None:
    linhas.append("</svg>")
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def coordenadas_x(resultados: Sequence[ResultadoResolucao], esquerda: float, largura: float) -> List[float]:
    if len(resultados) == 1:
        return [esquerda + largura / 2]
    passo = largura / (len(resultados) - 1)
    return [esquerda + indice * passo for indice in range(len(resultados))]


def desenhar_eixos(
    linhas: List[str], resultados: Sequence[ResultadoResolucao], xs: Sequence[float],
    esquerda: float, topo: float, largura: float, altura: float, maximo_y: float, unidade_y: str,
) -> None:
    base = topo + altura
    for indice in range(6):
        valor = maximo_y * indice / 5
        y = base - altura * indice / 5
        linhas.append(f'  <line class="grade" x1="{esquerda}" y1="{y:.1f}" x2="{esquerda + largura}" y2="{y:.1f}"/>')
        linhas.append(f'  <text class="pequeno" x="{esquerda - 14}" y="{y + 5:.1f}" text-anchor="end">{formatar_numero(valor)}</text>')
    linhas.append(f'  <line class="eixo" x1="{esquerda}" y1="{topo}" x2="{esquerda}" y2="{base}"/>')
    linhas.append(f'  <line class="eixo" x1="{esquerda}" y1="{base}" x2="{esquerda + largura}" y2="{base}"/>')
    for x, resultado in zip(xs, resultados):
        linhas.append(f'  <line class="eixo" x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base + 7}"/>')
        linhas.append(f'  <text class="pequeno" x="{x:.1f}" y="{base + 29}" text-anchor="middle">{resultado.tamanho} × {resultado.tamanho}</text>')
    linhas.append(f'  <text class="rotulo" x="{esquerda + largura / 2:.1f}" y="{base + 74}" text-anchor="middle">Resolução da imagem (pixels)</text>')
    linhas.append(f'  <text class="rotulo" transform="translate(29 {topo + altura / 2:.1f}) rotate(-90)" text-anchor="middle">{escapar(unidade_y)}</text>')


def pontos_serie(
    resultados: Sequence[ResultadoResolucao], xs: Sequence[float], valores: Sequence[Optional[float]],
    topo: float, altura: float, maximo_y: float,
) -> List[Tuple[float, float, float]]:
    pontos: List[Tuple[float, float, float]] = []
    for x, valor in zip(xs, valores):
        if valor is not None:
            y = topo + altura * (1 - valor / maximo_y)
            pontos.append((x, y, valor))
    return pontos


def coordenada_y_logaritmica(valor: float, minimo: float, maximo: float, topo: float, altura: float) -> float:
    """Converte um tempo positivo para a coordenada vertical em escala log10."""
    log_minimo = math.log10(minimo)
    log_maximo = math.log10(maximo)
    return topo + altura * (1 - (math.log10(valor) - log_minimo) / (log_maximo - log_minimo))


def desenhar_eixos_tempo_logaritmico(
    linhas: List[str], resultados: Sequence[ResultadoResolucao], xs: Sequence[float],
    esquerda: float, topo: float, largura: float, altura: float, minimo: float, maximo: float,
) -> None:
    """Desenha os eixos do gráfico de tempo sem esconder a série mais rápida."""
    base = topo + altura
    expoente_inicial = math.floor(math.log10(minimo))
    expoente_final = math.ceil(math.log10(maximo))
    for expoente in range(expoente_inicial, expoente_final + 1):
        valor = 10.0 ** expoente
        y = coordenada_y_logaritmica(valor, minimo, maximo, topo, altura)
        linhas.append(f'  <line class="grade" x1="{esquerda}" y1="{y:.1f}" x2="{esquerda + largura}" y2="{y:.1f}"/>')
        linhas.append(f'  <text class="pequeno" x="{esquerda - 14}" y="{y + 5:.1f}" text-anchor="end">{formatar_numero(valor)}</text>')
    linhas.append(f'  <line class="eixo" x1="{esquerda}" y1="{topo}" x2="{esquerda}" y2="{base}"/>')
    linhas.append(f'  <line class="eixo" x1="{esquerda}" y1="{base}" x2="{esquerda + largura}" y2="{base}"/>')
    for x, resultado in zip(xs, resultados):
        linhas.append(f'  <line class="eixo" x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base + 7}"/>')
        linhas.append(f'  <text class="pequeno" x="{x:.1f}" y="{base + 29}" text-anchor="middle">{resultado.tamanho} × {resultado.tamanho}</text>')
    linhas.append(f'  <text class="rotulo" x="{esquerda + largura / 2:.1f}" y="{base + 74}" text-anchor="middle">Resolução da imagem (pixels)</text>')
    linhas.append(f'  <text class="rotulo" transform="translate(29 {topo + altura / 2:.1f}) rotate(-90)" text-anchor="middle">Tempo médio (ms, escala logarítmica)</text>')


def pontos_serie_logaritmica(
    xs: Sequence[float], valores: Sequence[Optional[float]], minimo: float, maximo: float,
    topo: float, altura: float,
) -> List[Tuple[float, float, float]]:
    pontos: List[Tuple[float, float, float]] = []
    for x, valor in zip(xs, valores):
        if valor is not None and valor > 0:
            y = coordenada_y_logaritmica(valor, minimo, maximo, topo, altura)
            pontos.append((x, y, valor))
    return pontos


def desenhar_barras_erro_logaritmicas(
    linhas: List[str], xs: Sequence[float], valores: Sequence[Optional[float]], desvios: Sequence[Optional[float]],
    cor: str, minimo: float, maximo: float, topo: float, altura: float,
) -> None:
    """Mostra o desvio-padrão entre as cinco tentativas sem competir com as linhas."""
    for x, valor, desvio in zip(xs, valores, desvios):
        if valor is None or valor <= 0 or desvio is None or desvio <= 0:
            continue
        inferior = max(minimo, valor - desvio)
        superior = min(maximo, valor + desvio)
        y_inferior = coordenada_y_logaritmica(inferior, minimo, maximo, topo, altura)
        y_superior = coordenada_y_logaritmica(superior, minimo, maximo, topo, altura)
        linhas.append(f'  <line x1="{x:.1f}" y1="{y_inferior:.1f}" x2="{x:.1f}" y2="{y_superior:.1f}" stroke="{cor}" stroke-width="2" opacity="0.58"/>')
        linhas.append(f'  <line x1="{x - 5:.1f}" y1="{y_inferior:.1f}" x2="{x + 5:.1f}" y2="{y_inferior:.1f}" stroke="{cor}" stroke-width="2" opacity="0.58"/>')
        linhas.append(f'  <line x1="{x - 5:.1f}" y1="{y_superior:.1f}" x2="{x + 5:.1f}" y2="{y_superior:.1f}" stroke="{cor}" stroke-width="2" opacity="0.58"/>')


def desenhar_serie(
    linhas: List[str], pontos: Sequence[Tuple[float, float, float]], cor: str, rotulo: str,
    deslocamento_rotulo: float,
) -> None:
    if not pontos:
        return
    coordenadas = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pontos)
    linhas.append(f'  <polyline points="{coordenadas}" fill="none" stroke="{cor}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, y, _ in pontos:
        linhas.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#FFFFFF" stroke="{cor}" stroke-width="3"/>')
    x, y, _ = pontos[-1]
    linhas.append(f'  <text x="{x + 15:.1f}" y="{y + deslocamento_rotulo:.1f}" font-size="18" font-weight="700" style="fill: {cor};">{escapar(rotulo)}</text>')


def grafico_tempos(resultados: Sequence[ResultadoResolucao], destino: Path, tentativas: int, repeticoes: int) -> None:
    valores_cpu = [resultado.media_cpu for resultado in resultados]
    valores_gpu = [resultado.media_gpu if resultado.tem_medida_gpu_valida else None for resultado in resultados]
    desvios_cpu = [resultado.desvio_cpu for resultado in resultados]
    desvios_gpu = [resultado.desvio_gpu if resultado.tem_medida_gpu_valida else None for resultado in resultados]
    valores = [valor for valor in valores_cpu + valores_gpu if valor is not None and valor > 0]
    esquerda, topo, largura, altura = 130, 165, 880, 420
    xs = coordenadas_x(resultados, esquerda, largura)
    titulo = "Tempo médio de processamento por resolução"
    descricao = "Linhas para CPU e GPU em escala logarítmica. As barras verticais mostram um desvio-padrão entre cinco tentativas."
    linhas = cabecalho_svg(titulo, descricao)
    linhas.extend([
        f'  <text class="titulo" x="{esquerda}" y="53">{titulo}</text>',
        f'  <text class="subtitulo" x="{esquerda}" y="82">Escala logarítmica: revela CPU e GPU apesar de tempos em ordens de grandeza diferentes.</text>',
        f'  <text class="subtitulo" x="{esquerda}" y="108">Barras = ±1 desvio-padrão ({tentativas} tentativas; {repeticoes} repetições internas cada).</text>',
    ])
    if valores:
        minimo_y = 10.0 ** math.floor(math.log10(min(valores) * 0.8))
        maximo_y = 10.0 ** math.ceil(math.log10(max(valores) * 1.2))
        if minimo_y == maximo_y:
            maximo_y *= 10
        desenhar_eixos_tempo_logaritmico(linhas, resultados, xs, esquerda, topo, largura, altura, minimo_y, maximo_y)
        desenhar_barras_erro_logaritmicas(linhas, xs, valores_cpu, desvios_cpu, AZUL, minimo_y, maximo_y, topo, altura)
        desenhar_barras_erro_logaritmicas(linhas, xs, valores_gpu, desvios_gpu, LARANJA, minimo_y, maximo_y, topo, altura)
        desenhar_serie(linhas, pontos_serie_logaritmica(xs, valores_cpu, minimo_y, maximo_y, topo, altura), AZUL, "CPU", -12)
        desenhar_serie(linhas, pontos_serie_logaritmica(xs, valores_gpu, minimo_y, maximo_y, topo, altura), LARANJA, "GPU", 25)
    else:
        desenhar_eixos(linhas, resultados, xs, esquerda, topo, largura, altura, 1.0, "Tempo médio (ms)")
        linhas.append(f'  <text class="rotulo" x="{esquerda + largura / 2}" y="{topo + altura / 2}" text-anchor="middle">Sem medições válidas.</text>')
    for x, resultado in zip(xs, resultados):
        if not resultado.tem_medida_gpu_valida:
            estado = "GPU inválida" if resultado.saidas_identicas is False else "GPU não medida"
            linhas.append(f'  <text class="pequeno" x="{x:.1f}" y="{topo - 12}" text-anchor="middle">{estado}</text>')
    finalizar_svg(linhas, destino)


def grafico_speedup(resultados: Sequence[ResultadoResolucao], destino: Path) -> None:
    valores_speedup = [resultado.speedup if resultado.tem_medida_gpu_valida else None for resultado in resultados]
    valores = [valor for valor in valores_speedup if valor is not None]
    maximo_y = limite_superior_bonito(max(max(valores, default=0), 1.0) * 1.18)
    esquerda, topo, largura, altura = 130, 145, 880, 440
    xs = coordenadas_x(resultados, esquerda, largura)
    titulo = "Speedup da GPU em relação à CPU"
    descricao = "Linha do speedup CPU dividido por GPU. A referência 1 indica tempos iguais; valores maiores favorecem a GPU."
    linhas = cabecalho_svg(titulo, descricao)
    linhas.extend([
        f'  <text class="titulo" x="{esquerda}" y="53">{titulo}</text>',
        '  <text class="subtitulo" x="130" y="82">Speedup = tempo médio da CPU ÷ tempo médio da GPU. Valores acima de 1 favorecem a GPU.</text>',
    ])
    desenhar_eixos(linhas, resultados, xs, esquerda, topo, largura, altura, maximo_y, "Speedup (×)")
    y_referencia = topo + altura * (1 - 1 / maximo_y)
    linhas.append(f'  <line x1="{esquerda}" y1="{y_referencia:.1f}" x2="{esquerda + largura}" y2="{y_referencia:.1f}" stroke="#374151" stroke-width="2" stroke-dasharray="7 6"/>')
    linhas.append(f'  <text class="pequeno" x="{esquerda + largura - 5}" y="{y_referencia - 9:.1f}" text-anchor="end">referência: 1×</text>')
    pontos = pontos_serie(resultados, xs, valores_speedup, topo, altura, maximo_y)
    desenhar_serie(linhas, pontos, VERDE, "speedup", -14)
    for x, y, valor in pontos:
        linhas.append(f'  <text class="pequeno" x="{x:.1f}" y="{y - 13:.1f}" text-anchor="middle">{valor:.2f}×</text>')
    if not valores:
        linhas.append(f'  <text class="rotulo" x="{esquerda + largura / 2}" y="{topo + altura / 2}" text-anchor="middle">Sem pares CPU/GPU válidos.</text>')
    finalizar_svg(linhas, destino)


def celula_tempo(media: Optional[float], desvio: Optional[float], disponivel: bool = True) -> str:
    if not disponivel or media is None:
        return "—"
    if desvio is None:
        return f"{media:.6f}"
    return f"{media:.6f} ± {desvio:.6f}"


def escapar_markdown(valor: object) -> str:
    return str(valor).replace("|", "\\|").replace("\n", " ")


def escrever_relatorio(
    resultados: Sequence[ResultadoResolucao], ambiente: Dict[str, str], destino: Path,
    tentativas: int, repeticoes: int,
) -> None:
    linhas = [
        "# Resultados do experimento: Sobel sequencial versus CUDA",
        "",
        "Este relatório foi gerado automaticamente por `scripts/executar_experimento.py`; os valores não são estimativas nem resultados reutilizáveis de outra máquina.",
        "",
        "## Configuração da máquina",
        "",
        "| Item | Valor |",
        "|---|---|",
    ]
    nomes = {
        "sistema_operacional": "Sistema operacional",
        "cpu": "CPU",
        "ram": "Memória RAM",
        "gpu": "GPU / driver",
        "cuda_toolkit": "CUDA Toolkit (`nvcc --version`)",
        "compilador_c": "Compilador C",
        "python": "Python que executou o experimento",
        "gerado_em_utc": "Gerado em (UTC)",
    }
    for chave, rotulo in nomes.items():
        linhas.append(f"| {rotulo} | {escapar_markdown(ambiente.get(chave, 'Nao identificado'))} |")

    linhas.extend([
        "",
        "## Método",
        "",
        f"Foram solicitadas **{tentativas} tentativas independentes** por implementação e resolução. Cada execução informou a média de **{repeticoes} repetições internas**, após o aquecimento já implementado nos executáveis.",
        "",
        "A medida da CPU é somente o Sobel. A medida da GPU é o tempo no host de lançamento do kernel mais `cudaDeviceSynchronize()` por repetição; leitura, escrita e alocações ficam fora da região cronometrada. O speedup é `tempo médio CPU / tempo médio GPU`.",
        "",
        "## Resultados",
        "",
        "| Resolução | Tempo médio CPU (ms) | Tempo médio GPU (ms) | Speedup | Saídas idênticas? |",
        "|---:|---:|---:|---:|:---:|",
    ])
    for resultado in resultados:
        gpu_disponivel = resultado.tem_medida_gpu_valida
        if resultado.saidas_identicas is True:
            igualdade = "Sim"
        elif resultado.saidas_identicas is False:
            igualdade = "Não"
        else:
            igualdade = "Não medida"
        speedup = f"{resultado.speedup:.3f}×" if resultado.speedup is not None and gpu_disponivel else "—"
        if resultado.saidas_identicas is False:
            tempo_gpu = "Inválido"
        else:
            tempo_gpu = celula_tempo(resultado.media_gpu, resultado.desvio_gpu, gpu_disponivel)
        linhas.append(
            f"| {resultado.tamanho} × {resultado.tamanho} | "
            f"{celula_tempo(resultado.media_cpu, resultado.desvio_cpu)} | "
            f"{tempo_gpu} | "
            f"{speedup} | {igualdade} |"
        )

    falhas = [resultado for resultado in resultados if resultado.erro_gpu]
    if falhas:
        linhas.extend(["", "### Resoluções sem resultado da GPU", ""])
        for resultado in falhas:
            linhas.append(f"- **{resultado.tamanho} × {resultado.tamanho}:** {resultado.erro_gpu}")

    invalidos = [resultado for resultado in resultados if resultado.erro_validacao]
    if invalidos:
        linhas.extend(["", "### Medições invalidadas pela verificação de corretude", ""])
        for resultado in invalidos:
            linhas.append(f"- **{resultado.tamanho} × {resultado.tamanho}:** {resultado.erro_validacao}")

    linhas.extend([
        "",
        "## Gráficos",
        "",
        "Os gráficos usam cores distinguíveis e rótulos diretos nas séries. O gráfico de tempo usa escala logarítmica no eixo vertical, necessária para tornar visíveis simultaneamente os tempos de CPU e GPU, que estão em ordens de grandeza diferentes; as barras indicam ±1 desvio-padrão entre as tentativas. O gráfico de speedup mantém a linha tracejada de 1× como ponto de equilíbrio.",
        "",
        "![Tempo médio de processamento](tempo_medio.svg)",
        "",
        "![Speedup da GPU](speedup.svg)",
        "",
        "## Interpretação automática",
        "",
    ])
    validos = [resultado for resultado in resultados if resultado.tem_medida_gpu_valida and resultado.speedup is not None]
    mais_rapidos = [resultado for resultado in validos if resultado.speedup is not None and resultado.speedup > 1]
    if mais_rapidos:
        primeiro = min(mais_rapidos, key=lambda resultado: resultado.tamanho)
        linhas.append(
            f"O primeiro tamanho avaliado em que a GPU foi mais rápida foi **{primeiro.tamanho} × {primeiro.tamanho}** (speedup de **{primeiro.speedup:.3f}×**)."
        )
    elif validos:
        linhas.append("A GPU não foi mais rápida em nenhum dos tamanhos avaliados nesta execução.")
    else:
        linhas.append("Não há pares CPU/GPU válidos suficientes para concluir em que resolução a GPU se torna vantajosa.")
    linhas.extend([
        "",
        "As medições brutas estão em `medicoes_brutas.csv`; a saída completa de cada execução e a comparação das imagens estão em `logs/`. O desvio apresentado na tabela é o desvio-padrão entre as tentativas, não entre as repetições internas.",
        "",
    ])
    destino.write_text("\n".join(linhas), encoding="utf-8")


def main() -> int:
    argumentos = criar_argumentos().parse_args()
    tamanhos = sorted(set(argumentos.tamanhos))
    pasta_resultados = caminho_absoluto(argumentos.resultados)
    ambiente = coletar_ambiente()
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    (pasta_resultados / "ambiente.json").write_text(
        json.dumps(ambiente, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        if not argumentos.sem_compilar:
            print("Compilando executáveis com make...")
            compilar()
        executavel_cpu = localizar_executavel(argumentos.executavel_cpu, "seq")
        executavel_gpu = localizar_executavel(argumentos.executavel_gpu, "cuda")
        print(f"CPU: {executavel_cpu}")
        print(f"GPU: {executavel_gpu}")
        print(f"Resultados: {pasta_resultados}")
        resultados = executar_medicoes(
            tamanhos,
            executavel_cpu,
            executavel_gpu,
            argumentos.repeticoes,
            argumentos.tentativas,
            pasta_resultados,
            argumentos.continuar_apos_falha_gpu,
        )
    except ErroExperimento as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1

    grafico_tempos(resultados, pasta_resultados / "tempo_medio.svg", argumentos.tentativas, argumentos.repeticoes)
    grafico_speedup(resultados, pasta_resultados / "speedup.svg")
    escrever_relatorio(
        resultados, ambiente, pasta_resultados / "relatorio.md", argumentos.tentativas, argumentos.repeticoes
    )
    if any(resultado.saidas_identicas is False for resultado in resultados):
        print(
            "ERRO: há saídas CPU/GPU diferentes; tempos da GPU e speedups foram invalidados. "
            "Consulte os logs.",
            file=sys.stderr,
        )
        return 2

    falhas_gpu = [resultado for resultado in resultados if resultado.erro_gpu]
    if any("Controle de Aplicativo" in resultado.erro_gpu for resultado in falhas_gpu):
        print(
            "ERRO: o Windows bloqueou o executável CUDA; nenhuma medição da GPU é válida. "
            "Consulte o relatório e solicite a liberação à TI/ao administrador.",
            file=sys.stderr,
        )
        return 1
    if falhas_gpu:
        print("Experimento concluído parcialmente; consulte as resoluções sem resultado da GPU no relatório.")
        print(f"Relatório: {pasta_resultados / 'relatorio.md'}")
        return 0

    print("Experimento concluído com saídas CPU/GPU idênticas.")
    print(f"Relatório: {pasta_resultados / 'relatorio.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

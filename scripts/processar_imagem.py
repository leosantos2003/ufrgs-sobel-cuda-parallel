#!/usr/bin/env python3
"""Executa o Sobel para PNG/JPEG/TIFF e salva a visualizacao final em uma imagem."""
import argparse
from pathlib import Path
import subprocess
import tempfile

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise SystemExit(
        "Este script precisa do Pillow. Instale com: python3 -m pip install Pillow"
    )


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
DIRETORIO_SAIDA = RAIZ_PROJETO / "pics-out"
FORMATOS_ENTRADA = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pgm"}
FORMATOS_SAIDA = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


def salvar_pgm(caminho: Path, imagem: Image.Image) -> None:
    """Grava uma imagem em tons de cinza como PGM P5 de 8 bits."""
    cinza = imagem.convert("L")
    largura, altura = cinza.size
    caminho.write_bytes(
        f"P5\n{largura} {altura}\n255\n".encode("ascii") + cinza.tobytes()
    )


parser = argparse.ArgumentParser(
    description=(
        "Converte uma imagem para PGM, executa o Sobel e salva a visualizacao "
        "final em PNG, JPG ou JPEG."
    )
)
parser.add_argument(
    "entrada", type=Path, help="imagem PNG, JPG, JPEG, TIFF ou PGM"
)
parser.add_argument(
    "--modo", choices=("cpu", "gpu"), default="cpu", help="implementacao Sobel (padrao: cpu)"
)
parser.add_argument(
    "--repeticoes", type=int, default=10, help="numero de execucoes para a medicao (padrao: 10)"
)
parser.add_argument(
    "--saida",
    type=Path,
    help="nome do arquivo PNG/JPG/JPEG em pics-out/ (padrao: <entrada>_sobel.png)",
)
parser.add_argument(
    "--sobrescrever", action="store_true", help="permite substituir uma imagem de saida existente"
)
args = parser.parse_args()

entrada = args.entrada.resolve()
if not entrada.is_file():
    parser.error(f"imagem de entrada nao encontrada: {args.entrada}")
if entrada.suffix.lower() not in FORMATOS_ENTRADA:
    parser.error("a entrada deve ter extensao .png, .jpg, .jpeg, .tif, .tiff ou .pgm")
if not 1 <= args.repeticoes <= 1_000_000:
    parser.error("--repeticoes deve estar entre 1 e 1000000")

if args.saida is None:
    saida = DIRETORIO_SAIDA / f"{entrada.stem}_sobel.png"
else:
    if args.saida.is_absolute() or args.saida.parent != Path("."):
        parser.error("--saida deve informar somente um nome de arquivo; a saida e salva em pics-out/")
    saida = DIRETORIO_SAIDA / args.saida
if saida.suffix.lower() not in FORMATOS_SAIDA:
    parser.error("a saida deve ter extensao .png, .jpg ou .jpeg")
if saida.resolve() == entrada:
    parser.error("a saida deve ser diferente da imagem de entrada")
if saida.exists() and not args.sobrescrever:
    parser.error(f"a saida ja existe: {saida}. Use --sobrescrever para substitui-la")

executavel = RAIZ_PROJETO / ("sobel_cuda" if args.modo == "gpu" else "sobel_seq")
if not executavel.is_file():
    alvo = "gpu" if args.modo == "gpu" else "cpu"
    raise SystemExit(f"Executavel ausente. Compile primeiro com: make {alvo}")

try:
    with Image.open(entrada) as imagem:
        imagem.load()
        imagem_cinza = imagem.convert("L")
except (OSError, UnidentifiedImageError) as erro:
    raise SystemExit(f"Nao foi possivel abrir a imagem: {erro}")

try:
    DIRETORIO_SAIDA.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sobel-") as diretorio_temporario:
        diretorio = Path(diretorio_temporario)
        entrada_pgm = diretorio / "entrada.pgm"
        saida_pgm = diretorio / "saida.pgm"
        salvar_pgm(entrada_pgm, imagem_cinza)

        subprocess.run(
            [str(executavel), str(entrada_pgm), str(saida_pgm), str(args.repeticoes)],
            check=True,
        )

        with Image.open(saida_pgm) as resultado:
            resultado.load()
            resultado.convert("L").save(saida, format=FORMATOS_SAIDA[saida.suffix.lower()])
except subprocess.CalledProcessError as erro:
    raise SystemExit(f"A execucao do Sobel falhou (codigo {erro.returncode}).")
except OSError as erro:
    raise SystemExit(f"Falha ao processar os arquivos: {erro}")

print(f"Imagem com bordas salva em: {saida}")

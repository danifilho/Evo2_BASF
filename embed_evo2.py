#!/usr/bin/env python3
"""
embed_evo2.py – gera embeddings Evo 2 para todas as janelas FASTA em um diretório.

Uso básico (GPU):
    python embed_evo2.py \
        --indir fasta_windows/ \
        --outdir embeddings_full \
        --model evo2_7b \
        --layer blocks.28.mlp.l3 \
        --recursive \
        --hf_cache ./huggingface        # opcional: onde salvar os pesos
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import List

import torch
from evo2 import Evo2
from tqdm import tqdm


# Command line arguments
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed every FASTA window in a directory with Evo 2."
    )
    p.add_argument(
        "--indir",
        required=True,
        type=pathlib.Path,
        help="Diretório com *.fa / *.fasta (janelas).",
    )
    p.add_argument(
        "--outdir",
        default=pathlib.Path("embeddings_full"),
        type=pathlib.Path,
        help="Para onde escrever os tensores .pt (criado se não existir).",
    )
    p.add_argument(
        "--layer",
        default="blocks.28.mlp.l3",
        help="Nome da camada a extrair (paper cita blocks.27.mlp.l3).",
    )
    p.add_argument(
        "--model",
        default="evo2_7b",
        help="Checkpoint Evo 2 a carregar (evo2_7b, evo2_40b, etc.).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Explorar subdiretórios dentro de --indir.",
    )
    p.add_argument(
        "--hf_cache",
        type=pathlib.Path,
        help="Caminho para cache do HuggingFace (opcional).",
    )
    return p.parse_args()

def get_fasta_paths(indir: pathlib.Path, recursive: bool) -> List[pathlib.Path]:
    pattern = "**/*.fa*" if recursive else "*.fa*"
    return sorted(indir.glob(pattern))


def load_sequence(path):
    return path.read_text().splitlines()[0].strip().upper()

# Main function
def main() -> None:
    args = parse_args()

    # Configura cache HuggingFace, se solicitado
    if args.hf_cache:
        os.environ["HF_HOME"] = str(args.hf_cache)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> usando dispositivo: {device}")

    # Instancia o modelo diretamente no device desejado
    t0 = time.time() 
    evo2_model = Evo2(args.model)        # sem device=
    evo2_model.model.to(device)          # move pesos p/ GPU
    evo2_model.model.eval()              # modo inferência

    tok = evo2_model.tokenizer

    print(f"-> modelo {args.model} carregado em {time.time() - t0:.1f}s")

    # Descobre arquivos FASTA
    fasta_paths = get_fasta_paths(args.indir, args.recursive)
    if not fasta_paths:
        sys.exit(f"Nenhum FASTA encontrado em {args.indir}")
    args.outdir.mkdir(exist_ok=True, parents=True)

    # Loop principal
    for fa_path in tqdm(fasta_paths, desc="Embeddings", unit="file"):
        out_file = args.outdir / f"{fa_path.stem}.pt"
        if out_file.exists():
            tqdm.write(f"→ {out_file.name} já existe; pulando.")
            continue

        sequence = load_sequence(fa_path)
        input_ids = torch.tensor(tok.tokenize(sequence), dtype=torch.int, device=device).unsqueeze(0)

        with torch.inference_mode():
            _, embeddings = evo2_model(
                input_ids, return_embeddings=True, layer_names=[args.layer]
            )

        torch.save(embeddings[args.layer].squeeze(0).cpu(), out_file)
        tqdm.write(f"✓ salvo {out_file.name}")

    print(
        f"\nConcluído – {len(fasta_paths)} arquivos em {time.time() - t0:.1f}s. "
        f"Tensores em {args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()


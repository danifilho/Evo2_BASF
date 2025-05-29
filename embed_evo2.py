#!/usr/bin/env python3
"""
embed_evo2_dir.py – create Evo2 embeddings for *all* FASTA windows in a folder
"""

import argparse
import pathlib
import sys
import time
import torch
from evo2 import Evo2
from tqdm import tqdm   # tiny progress bar; pip install tqdm if needed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed every FASTA window in a directory with Evo2."
    )
    p.add_argument(
        "--indir",
        required=True,
        type=pathlib.Path,
        help="Directory containing *.fa / *.fasta windows.",
    )
    p.add_argument(
        "--outdir",
        default=pathlib.Path("embeddings_full"),
        type=pathlib.Path,
        help="Where .pt tensors will be written (created if missing).",
    )
    p.add_argument(
        "--layer",
        default="blocks.28.mlp.l3",
        help="Evo2 layer name for embeddings (paper notes blocks.27.mlp.l3).",
    )
    p.add_argument(
        "--model",
        default="evo2_7b",
        help="Which Evo2 checkpoint to load (evo2_7b, evo2_15b-fp16, …).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into sub-directories of --indir.",
    )
    return p.parse_args()


def get_fasta_paths(indir: pathlib.Path, recursive: bool) -> list[pathlib.Path]:
    pattern = "**/*.fa*" if recursive else "*.fa*"
    return sorted(indir.glob(pattern))


def load_sequence(path: pathlib.Path) -> str:
    # FASTA windows are one-liner after the header, so grab the 2nd line only
    return path.read_text().splitlines()[0].strip().upper()


def main() -> None:
    args = parse_args()

    # 1. Model
    t0 = time.time()
    evo2_model = Evo2(args.model)
    tok = evo2_model.tokenizer

    # 2. File list
    fasta_paths = get_fasta_paths(args.indir, args.recursive)
    if not fasta_paths:
        sys.exit(f"No FASTA files found in {args.indir}")

    args.outdir.mkdir(exist_ok=True, parents=True)

    # 3. Iterate
    for fa_path in tqdm(fasta_paths, desc="Embeddings", unit="file"):
        out_file = args.outdir / (fa_path.stem + ".pt")
        if out_file.exists():
            # Optional: skip to avoid recomputing
            tqdm.write(f"→ {out_file.name} already exists; skipping.")
            continue

        sequence = load_sequence(fa_path)
        input_ids = torch.tensor(tok.tokenize(sequence), dtype=torch.int)[None]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.inference_mode():
            _, embeddings = evo2_model(
                input_ids, return_embeddings=True, layer_names=[args.layer]
            )

        torch.save(embeddings[args.layer].squeeze(0).cpu(), out_file)
        tqdm.write(f"✓ saved {out_file.name}")

    print(
        f"\nDone – processed {len(fasta_paths)} files in {time.time() - t0:.1f}s, "
        f"wrote tensors to {args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import joblib

from src.extract import load_model_and_tokenizer, extract_hidden_states_word_level


def clean_line(line: str) -> str:
    line = line.strip()
    # remove optional language prefix like [EN] or [FA]
    if line.startswith("[") and "]" in line[:5]:
        line = line.split("]", 1)[1].strip()
    return line


def main():
    ap = argparse.ArgumentParser(description="Use a trained probe to POS-tag tokens in a text file.")
    ap.add_argument("--text_file", type=str, required=True, help="Path to a text file (one sentence per line)")
    ap.add_argument("--probe_ckpt", type=str, required=True, help="Path to probes.joblib (from run_probe.py --save_probe)")
    ap.add_argument("--layer", type=int, default=6, help="Which layer's probe to use")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_csv", type=str, default="demo/outputs/tagged.csv")
    args = ap.parse_args()

    ckpt = joblib.load(args.probe_ckpt)
    probes = ckpt["probes"]
    le = ckpt["label_encoder"]
    model_name = ckpt.get("model", "bert-base-multilingual-cased")

    if args.layer not in probes:
        raise ValueError(f"Layer {args.layer} not found in checkpoint. Available: {sorted(probes.keys())}")

    tokenizer, model = load_model_and_tokenizer(model_name, device=args.device)
    probe = probes[args.layer]

    text_path = Path(args.text_file)
    lines = [clean_line(x) for x in text_path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.strip().startswith("#")]

    rows = []
    for i, line in enumerate(lines):
        tokens = line.split()
        ext = extract_hidden_states_word_level(
            tokens, tokenizer, model, layers=[args.layer], device=args.device, return_attentions=False
        )
        X = ext.hidden_by_layer[args.layer]
        pred_ids = probe.predict(X)
        pred_tags = le.inverse_transform(pred_ids)

        for tok, tag in zip(ext.word_tokens, pred_tags):
            rows.append({"line_id": i, "token": tok, "pred_upos": tag})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved -> {out_csv}")


if __name__ == "__main__":
    main()

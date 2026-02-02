from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import joblib

from src.ud import load_ud, flatten_tokens_and_labels
from src.extract import load_model_and_tokenizer
from src.probe import build_token_level_matrices, train_and_eval_layerwise_probes


def parse_layers(s: str) -> List[int]:
    """Parse layers from formats like:
    - '0-12'
    - '0,1,2,3'
    - '0-12,24' (mixed)
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty layers string")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    layers: List[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-")
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            layers.extend(list(range(a, b + step, step)))
        else:
            layers.append(int(p))
    # unique while preserving order
    seen = set()
    out = []
    for x in layers:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser(description="Layer-wise UPOS probing prototype")
    ap.add_argument("--ud_config", type=str, default="en_ewt", help="UD config, e.g., en_ewt, fa_seraji")
    ap.add_argument("--model", type=str, default="bert-base-multilingual-cased")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--layers", type=str, default="0-12")
    ap.add_argument("--max_sentences", type=int, default=200, help="Max sentences per split (train/eval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="demo/outputs/probe_results.csv")
    ap.add_argument("--save_probe", action="store_true", help="Save trained probes + label encoder to joblib")
    ap.add_argument("--save_dir", type=str, default="demo/outputs")
    ap.add_argument("--keep_punct", action="store_true", help="Do not drop UPOS=PUNCT")
    args = ap.parse_args()

    layers = parse_layers(args.layers)

    print(f"[INFO] Loading UD: {args.ud_config}")
    train_sents = load_ud(args.ud_config, split="train", max_sentences=args.max_sentences, seed=args.seed)

    # Prefer validation if available; else test
    try:
        eval_sents = load_ud(args.ud_config, split="validation", max_sentences=args.max_sentences, seed=args.seed)
        if len(eval_sents) == 0:
            raise ValueError("Empty validation")
    except Exception:
        eval_sents = load_ud(args.ud_config, split="test", max_sentences=args.max_sentences, seed=args.seed)

    train_tokens, train_upos = flatten_tokens_and_labels(train_sents)
    eval_tokens, eval_upos = flatten_tokens_and_labels(eval_sents)

    print(f"[INFO] Loading model: {args.model} ({args.device})")
    tokenizer, model = load_model_and_tokenizer(args.model, device=args.device)

    drop = None if args.keep_punct else ("PUNCT",)

    print("[INFO] Extracting train matrices...")
    Xtr_by_layer, ytr = build_token_level_matrices(
        train_tokens, train_upos, tokenizer, model, layers=layers, device=args.device, drop_upos=drop
    )
    print("[INFO] Extracting eval matrices...")
    Xev_by_layer, yev = build_token_level_matrices(
        eval_tokens, eval_upos, tokenizer, model, layers=layers, device=args.device, drop_upos=drop
    )

    print("[INFO] Training + evaluating probes...")
    results, probes, le = train_and_eval_layerwise_probes(
        Xtr_by_layer, ytr, Xev_by_layer, yev, seed=args.seed
    )

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("layer")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved CSV -> {out_csv}")

    if args.save_probe:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"probes": probes, "label_encoder": le, "layers": layers, "model": args.model}, save_dir / "probes.joblib")
        print(f"[OK] Saved probes -> {save_dir / 'probes.joblib'}")

    print(df)


if __name__ == "__main__":
    main()

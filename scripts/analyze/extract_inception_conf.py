#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-image InceptionV3 predictive entropy for every arm under a run
root (P1 offline gates; see .claude/covr_quality_constrained_budget_20260820.md).

The confidence definition is bit-identical to the one that passed the [a]
validity gate in ``scripts/analyze_crossover.py --metric inception_conf``:
predictive entropy (nats) of torch-fidelity InceptionV3 ``logits_unbiased``,
smaller = more confident.

Walks ``<root>/**/generated/*.png`` (the layout written by
``scripts/sweep_budget_probe.sh`` / static-mask sweeps: every arm run dir has
its own ``generated/``). Arm label = run dir path relative to root. Class name
is parsed from the ``{idx:06d}_{class_name}.png`` filename convention.

Output CSV columns: arm,image,class_name,conf
(plus ``budget`` parsed from a leading ``k<K>/`` path segment when present).

Analysis only — reads PNGs, never writes images. Run on the GPU box (needs
torch + torch-fidelity + cached weights).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.analyze.analyze_crossover import _entropy_from_logits, _inception_features  # noqa: E402

_PNG_NAME = re.compile(r"^(\d{6})_(.+)\.png$")


def _load_png_uint8(path):
    import torch  # noqa: F401  (environment guard)
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("out_csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    pattern = os.path.join(args.run_root, "**", "generated", "*.png")
    pngs = sorted(glob.glob(pattern, recursive=True))
    if not pngs:
        print(f"no PNGs under {pattern}")
        return 1
    print(f"{len(pngs)} PNGs under {args.run_root}")

    rows = []
    batch_paths, batch_imgs = [], []

    def flush():
        if not batch_imgs:
            return
        logits = _inception_features(batch_imgs, "logits_unbiased",
                                     args.device, batch=args.batch)
        ent = _entropy_from_logits(logits)
        for p, e in zip(batch_paths, ent):
            rows.append((p, float(e)))
        batch_paths.clear()
        batch_imgs.clear()

    for i, path in enumerate(pngs):
        batch_paths.append(path)
        batch_imgs.append(_load_png_uint8(path))
        if len(batch_imgs) >= args.batch:
            flush()
            if (i + 1) % 1000 < args.batch:
                print(f"  {i + 1}/{len(pngs)}")
    flush()

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "budget", "image", "class_name", "conf"])
        for path, ent in rows:
            rel = os.path.relpath(os.path.dirname(path), args.run_root)
            rel = os.path.dirname(rel)  # strip trailing "generated"
            m = _PNG_NAME.match(os.path.basename(path))
            name, cls = (m.group(1) + ".png", m.group(2)) if m else (
                os.path.basename(path), "")
            km = re.search(r"(?:^|/)(k(\d+))(?:/|$)", rel + "/")
            budget = km.group(2) if km else ""
            w.writerow([rel, budget, name, cls, f"{ent:.6f}"])
    print(f"wrote {len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

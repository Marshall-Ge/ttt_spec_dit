#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract InceptionV3 2048-pool features for the mini-FID sentinel gate
[P1'-a] (spec: .claude/covr_minifid_sentinel_20260821.md §1).

For every arm run dir under <root> (any dir owning a ``generated/`` with
PNGs) this extracts:

  * fake features: all ``generated/*.png``
  * real features: the sibling ``real_299/*.png`` — the EXACT real side the
    arm's own results.json FID was computed against (deduplicated across
    arms that share the same file list, e.g. paired arms within an offset)

Faithfulness to the runs' FID numbers:
  * extractor = torch-fidelity ``inception-v3-compat`` ``"2048"`` features
    via analyze_crossover._inception_features — the same network+weights
    torch-fidelity used for results.json;
  * preprocessing = lossless PNG -> uint8 -> PIL BICUBIC 299x299, bit-
    identical to eval/fid_is.py add() / compute_mixture_fid.py (no-op for
    real_299 which is already 299).

Output: one .npz with
    fake::<arm>        (n, 2048) float32
    fake_names::<arm>  (n,) filenames (sorted; subsampling reproducibility)
    real::<gid>        (m, 2048) float32   (gid = r0, r1, ... dedup groups)
    real_of            json str {arm: gid}
    meta               json str (root, feature, preprocessing, date)

Analysis only — reads PNGs, never writes images. Run on the GPU box.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_crossover import _inception_features  # noqa: E402


def _load_299_uint8(path):
    """Bit-identical to eval/fid_is.py add(): uint8 -> BICUBIC 299."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size != (299, 299):
        img = img.resize((299, 299), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _extract(paths, device, batch):
    feats = []
    for s in range(0, len(paths), batch):
        imgs = [_load_299_uint8(p) for p in paths[s:s + batch]]
        feats.append(_inception_features(imgs, "2048", device, batch=batch)
                     .astype(np.float32))
    return np.concatenate(feats, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("out_npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--real-dirname", default="real_299",
                    help="sibling real dir name next to generated/")
    args = ap.parse_args(argv)

    gen_dirs = sorted(
        os.path.dirname(p) for p in glob.glob(
            os.path.join(args.run_root, "**", "generated"), recursive=True)
        if os.path.isdir(p))
    arms = {}
    for gd in gen_dirs:
        pngs = sorted(glob.glob(os.path.join(gd, "generated", "*.png")))
        if pngs:
            arms[os.path.relpath(gd, args.run_root)] = pngs
    if not arms:
        print(f"no <run>/generated/*.png under {args.run_root}")
        return 1
    n_fake = sum(len(v) for v in arms.values())
    print(f"{len(arms)} arms, {n_fake} fake PNGs under {args.run_root}")

    out = {}
    real_of = {}
    real_groups = {}          # dedup key -> gid
    t0 = time.time()

    for i, (arm, pngs) in enumerate(sorted(arms.items())):
        out[f"fake::{arm}"] = _extract(pngs, args.device, args.batch)
        out[f"fake_names::{arm}"] = np.array(
            [os.path.basename(p) for p in pngs])

        rdir = os.path.join(args.run_root, arm, args.real_dirname)
        rpngs = sorted(glob.glob(os.path.join(rdir, "*.png")))
        if rpngs:
            # arms sharing an identical file list (paired arms within an
            # offset) share one real feature block; (name, size) pairs keep
            # same-named-but-different-content real sets apart
            key = tuple((os.path.basename(p), os.path.getsize(p))
                        for p in rpngs)
            if key not in real_groups:
                gid = f"r{len(real_groups)}"
                real_groups[key] = gid
                out[f"real::{gid}"] = _extract(rpngs, args.device, args.batch)
            real_of[arm] = real_groups[key]
        else:
            print(f"  WARNING: no {args.real_dirname}/ for {arm}")
        print(f"  [{i + 1}/{len(arms)}] {arm}: fake={len(pngs)} "
              f"real_gid={real_of.get(arm, '-')} "
              f"({time.time() - t0:.0f}s elapsed)")

    out["real_of"] = np.array(json.dumps(real_of))
    out["meta"] = np.array(json.dumps({
        "run_root": os.path.abspath(args.run_root),
        "feature": "inception-v3-compat/2048",
        "preprocess": "PNG->uint8->PIL BICUBIC 299 (eval/fid_is.py add())",
        "real_dirname": args.real_dirname,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }))
    np.savez_compressed(args.out_npz, **out)
    size_mb = os.path.getsize(args.out_npz) / 1e6
    print(f"wrote {args.out_npz} ({size_mb:.0f} MB, "
          f"{len(arms)} arms, {len(real_groups)} real groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

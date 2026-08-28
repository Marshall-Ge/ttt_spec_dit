# -*- coding: utf-8 -*-
"""Miscellaneous pure helpers without a dedicated domain module yet.

Per project-structure.md §5: new pure logic should land in the right
submodule; use this file only for small helpers that do not (yet) justify
their own module.
"""

import os
import re


def latent_seed_for_index(absolute_idx: int,
                          latent_seed_offset: int = 0) -> int:
    """Deterministic per-image latent seed.

    The (image, latent draw) cell is seeded by this formula so replicate runs
    over the SAME images can draw independent latents per image:

    - ``offset=0`` reproduces the legacy ``100000 + absolute_idx`` formula
      bit-identically (every existing run / cached comparison depends on it);
    - distinct offsets give disjoint seed sets for the same image indices
      (stride 1_000_000 > 50_000, the largest possible image index);
    - the same offset is reproducible across calls.

    Load-bearing invariants are pinned by ``tests/test_latent_seed_offset.py``.
    """
    if latent_seed_offset < 0:
        raise ValueError(
            f"latent_seed_offset must be >= 0, got {latent_seed_offset}")
    return 100000 + latent_seed_offset * 1_000_000 + int(absolute_idx)


def get_vfl_checkpoint_dir(output_dir: str, method: str) -> str:
    """VFL checkpoint directory under an experiment output dir."""
    return os.path.join(output_dir, f"vfl_checkpoints_{method}")


def prune_checkpoints(checkpoint_dir: str, keep: int = 3) -> int:
    """Delete oldest ``*.pt`` checkpoints in ``checkpoint_dir`` beyond ``keep``.

    Files are ordered by an embedded ``v<digits>`` version when present,
    otherwise lexicographically. Returns the number of removed files.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    if not os.path.isdir(checkpoint_dir):
        return 0

    def _version(name: str) -> int:
        m = re.search(r"v(\d+)", name)
        return int(m.group(1)) if m else 0

    files = sorted(
        (f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")),
        key=_version,
    )
    removed = 0
    doomed = files[:-keep] if keep > 0 else files
    for name in doomed:
        try:
            os.remove(os.path.join(checkpoint_dir, name))
            removed += 1
        except OSError:
            pass
    return removed

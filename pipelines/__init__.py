# -*- coding: utf-8 -*-
"""编排层 — 生成器与采样循环(见 .claude/project-structure.md §2)。

- dit.py    DiTGenerator(类条件,含 TTT 训练路径)
- pixart.py PixArtGenerator(t2i/c2i,T5 编码)
- hooks/    采样循环扩展钩子(COVR 等)
"""

from .dit import DiTGenerator
from .pixart import PixArtGenerator

__all__ = ["DiTGenerator", "PixArtGenerator"]

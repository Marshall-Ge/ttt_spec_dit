---
name: timestep feedback shadow integration smoke
description: 2026-08-18 first real DiT SpecA shadow integration smoke; selective timestep feedback state persisted without changing the main trajectory
type: project
---

Configuration: DiT/ImageNet, SpecA, 50 steps, CFG 1.0, 16 images, batch size 4,
feedback budget 8, `p_min=0.02`, beta 1.0, safety and sentinel rates 0.

The real sampling loop completed successfully:

- 4 trajectories;
- 25 selective timestep feedback contexts;
- 100 sample labels;
- state persisted to `covr/state.json`;
- SpecA full steps 48, Taylor steps 152, skip ratio 0.76;
- accelerated FLOPs 3.0698 T and vanilla FLOPs 11.8667 T.

This validates runtime wiring, selective label recording, and state persistence.
The smoke is not a quality result and is too small to establish online learning
benefit or trajectory parity beyond successful execution.

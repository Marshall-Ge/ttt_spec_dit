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

Two gotchas found while enabling the active mask (`--covr-timestep-feedback-active`):

1. **NaN crash from unconstrained risk masks**: the first active attempt spent
   the whole budget on high-risk early steps (1-10), leaving a ~24-step Taylor
   gap in the second half; SpecA diverged and `transition_defect_batch` raised
   "transition metrics must be finite and non-negative". Fix: gap safety must
   be enforced BEFORE risk-driven selection (`recommended_refresh_mask` phase
   1 repairs all gaps > max_taylor_gap, phase 2 spends leftover budget on risk).
2. **No adaptation slack at the minimum budget**: with 50 steps and
   max_taylor_gap=4 the minimum safe refresh count is 10 (ceil(50/5)); at
   budget=10 the mask is forced to the even layout [0,5,...,45] and the learned
   risk cannot influence placement. Risk adaptation needs budget strictly
   above the minimum (e.g. 12-14) to have slack.

Active deployment smoke (session s3, seed 44, budget 10) then ran clean:
40/200 full steps (exactly 10 refreshes per trajectory), skip ratio 0.80,
FLOPs speedup 5.0x, no NaN, 31 new selective contexts. This validates the
active-mask wiring only — at the minimum safe budget the mask is structurally
uniform, so no quality or adaptation claim can be made from it.

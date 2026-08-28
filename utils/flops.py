# -*- coding: utf-8 -*-
"""FLOPs profiling utilities — tail-only forward profilers.

These measure the FLOPs of the output tail (norm_out + proj_out + unpatchify)
by running a forward that skips all transformer blocks. Used by
``eval/latency.py``'s FLOPsMetric to estimate the cost of a TeaCache skip step.
"""

import torch

# ---------------------------------------------------------------------------
# Tail profilers (PixArt / DiT)
# ---------------------------------------------------------------------------

def _profile_tail_flops(transformer, latent_input, prompt_embeds, attn_mask,
                        current_t, added, device, dtype):
    """Run a forward that skips all transformer blocks, counting only the tail."""
    from torch.utils.flop_counter import FlopCounterMode

    batch_size = latent_input.shape[0]
    height = latent_input.shape[-2] // transformer.config.patch_size
    width = latent_input.shape[-1] // transformer.config.patch_size

    # Replicate the stock forward, but skip the block loop
    def tail_only_forward(hidden_states):
        """Only run pos_embed + adaln_single + tail (no blocks)."""
        hidden_states = transformer.pos_embed(hidden_states)

        timestep_emb, embedded_timestep = transformer.adaln_single(
            current_t, added, batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if transformer.caption_projection is not None:
            eh = transformer.caption_projection(prompt_embeds)
            eh = eh.view(batch_size, -1, hidden_states.shape[-1])
        else:
            eh = prompt_embeds

        # ── BLOCKS SKIPPED ──
        # hidden_states stays unchanged except for a tiny add (modeled below)

        # Step counter: advance to a non-zero state so the decision logic
        # doesn't interfere (we're profiling, not caching).

        # Tail (always runs)
        shift, scale = (
            transformer.scale_shift_table[None]
            + embedded_timestep[:, None].to(transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = (hidden_states * (1 + scale.to(hidden_states.device))
                         + shift.to(hidden_states.device))
        hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.config.patch_size,
                   transformer.config.patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels,
                   height * transformer.config.patch_size,
                   width * transformer.config.patch_size)
        )
        return output

    with FlopCounterMode(display=False) as fcm:
        _ = tail_only_forward(latent_input)
    return fcm.get_total_flops()


def _profile_tail_flops_dit(transformer, latent_input, current_t, class_labels,
                             device, dtype):
    """Run a forward that skips all DiT transformer blocks, counting only the tail."""
    from torch.utils.flop_counter import FlopCounterMode
    from torch.nn.functional import silu

    height = latent_input.shape[-2] // transformer.patch_size
    width = latent_input.shape[-1] // transformer.patch_size

    def tail_only_forward(hidden_states):
        """Only run pos_embed + tail for DiT (no blocks)."""
        hidden_states = transformer.pos_embed(hidden_states)

        # ── BLOCKS SKIPPED ──

        # DiT tail
        conditioning = transformer.transformer_blocks[0].norm1.emb(
            current_t, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = transformer.proj_out_1(silu(conditioning)).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = transformer.proj_out_2(hidden_states)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.patch_size,
                   transformer.patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels,
                   height * transformer.patch_size, width * transformer.patch_size)
        )
        return output

    with FlopCounterMode(display=False) as fcm:
        _ = tail_only_forward(latent_input)
    return fcm.get_total_flops()

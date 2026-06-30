# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic helpers for the Whisper-on-ROCm V2 NaN investigation.

Everything here is gated behind env vars and is a no-op unless enabled:

  VLLM_WHISPER_DEBUG=1        dump per-attention metadata (shape/dtype/stride/
                              contiguity/nan/inf) and a per-layer output NaN probe.
  VLLM_WHISPER_FORCE_CONTIG=1 force the read-side index/metadata tensors
                              (query_start_loc, seq_lens, block_table) contiguous
                              right before each kernel launch. Set together with
                              VLLM_WHISPER_DEBUG to confirm a contiguity fix.

Apply with `git apply`, run the failing test, then revert with `git apply -R`.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

WHISPER_DEBUG = int(os.environ.get("VLLM_WHISPER_DEBUG", "0"))
FORCE_CONTIG = int(os.environ.get("VLLM_WHISPER_FORCE_CONTIG", "0"))


def _fmt(name: str, t: object) -> str:
    if t is None:
        return f"{name}=None"
    if isinstance(t, torch.Tensor):
        try:
            cont = t.is_contiguous()
        except Exception:
            cont = "?"
        s = (
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"stride={tuple(t.stride())} contig={cont} dev={t.device}"
        )
        if t.numel() and t.is_floating_point():
            s += f" nan={bool(torch.isnan(t).any())} inf={bool(torch.isinf(t).any())}"
        elif t.numel():
            s += f" min={int(t.min())} max={int(t.max())}"
        return s
    return f"{name}={t!r}"


def dump_meta(tag: str, **tensors: object) -> None:
    if not WHISPER_DEBUG:
        return
    logger.info(
        "[whisper-dbg] %s | %s", tag, " | ".join(_fmt(n, v) for n, v in tensors.items())
    )


def maybe_contig(t: object) -> object:
    """Return a contiguous copy when FORCE_CONTIG is set; else passthrough."""
    if (
        FORCE_CONTIG
        and isinstance(t, torch.Tensor)
        and t.numel()
        and not t.is_contiguous()
    ):
        return t.contiguous()
    return t


def check_output_nan(layer_name: str, attn_type: str, output: object) -> None:
    """Per-layer probe: flag NaN/Inf in an attention layer's output."""
    if not WHISPER_DEBUG:
        return
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        return
    has_nan = bool(torch.isnan(output).any())
    has_inf = bool(torch.isinf(output).any())
    if has_nan or has_inf:
        logger.warning(
            "[whisper-dbg] NaN/Inf in attn OUTPUT: layer=%s type=%s "
            "nan=%s inf=%s shape=%s",
            layer_name,
            attn_type,
            has_nan,
            has_inf,
            tuple(output.shape),
        )
    else:
        logger.info(
            "[whisper-dbg] ok attn output: layer=%s type=%s amax=%.4g shape=%s",
            layer_name,
            attn_type,
            float(output.abs().max()),
            tuple(output.shape),
        )

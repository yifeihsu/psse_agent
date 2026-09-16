"""Masked independent binary supervision for WLS-only graph screening."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_bce(logits: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    target = torch.as_tensor(target, device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError(f"Target shape {target.shape} does not match logits {logits.shape}")
    if not torch.isfinite(logits).all():
        raise ValueError("Loss requires finite logits")
    if mask is None:
        usable = torch.ones_like(logits, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, device=logits.device)
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Supervision masks must be boolean or binary")
        try:
            usable = torch.broadcast_to(mask.bool(), logits.shape)
        except RuntimeError as exc:
            raise ValueError("Supervision mask is not broadcastable to its logits") from exc
    known = target[usable]
    if not torch.isfinite(known).all() or ((known < 0) | (known > 1)).any():
        raise ValueError("Available binary labels must be finite and lie in [0,1]")
    # Unknown labels may be NaN; neutralize them before BCE to keep gradients finite.
    safe_target = torch.where(usable, target, torch.zeros_like(target))
    element_loss = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    return (element_loss * usable).sum() / usable.sum().clamp_min(1)


def screen_loss(
    outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor], *,
    phase_weight: float = 1.0, anomaly_weight: float = 0.25, family_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Compute the guide's loss, averaging family BCE over available labels.

    Targets are ``phase`` [B], ``anomaly`` [B], and ``family`` [B,5]. Optional
    ``phase_mask``, ``anomaly_mask`` and ``family_mask`` exclude unknown labels.
    A family mask [5] can disable an entire unrepresented head. Phase and anomaly
    labels are explicit: incomplete family labels must not imply negative labels.
    Mixed cases use multiple positive family entries, never a class softmax.
    An entirely masked term is a differentiable zero. There is no localization or
    healthy-power-flow reconstruction loss.
    """
    if any(not math.isfinite(weight) or weight < 0 for weight in (phase_weight, anomaly_weight, family_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    phase_loss = _masked_bce(outputs["phase_screen_logit"], targets["phase"], targets.get("phase_mask"))
    anomaly_loss = _masked_bce(outputs["anomaly_logit"], targets["anomaly"], targets.get("anomaly_mask"))
    family_loss = _masked_bce(outputs["family_logits"], targets["family"], targets.get("family_mask"))
    loss = phase_weight * phase_loss + anomaly_weight * anomaly_loss + family_weight * family_loss
    return {"loss": loss, "phase_loss": phase_loss, "anomaly_loss": anomaly_loss, "family_loss": family_loss}

"""The digest of a prompt as the model saw it: the token ids fed to ``generate``.

The sampler records it with every generation and the row builder recomputes
it from the re-rendered prompt, so a training row is trained on exactly the
context the action was sampled from, not merely one of the same length
(``psse_env.rl.rows``).  Plain hashlib so both sides can import it.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

PROMPT_DIGEST_CONTRACT = "sha256 over the comma-joined decimal token ids"


def prompt_token_digest(token_ids: Iterable[int]) -> str:
    """Hex digest of the token id sequence under :data:`PROMPT_DIGEST_CONTRACT`."""

    text = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


__all__ = ["PROMPT_DIGEST_CONTRACT", "prompt_token_digest"]

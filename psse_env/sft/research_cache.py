"""Pre-cache and verify exact research model snapshots before GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from psse_env.research_models import GEMMA4_12B, GEMMA4_E4B, get_research_model_spec

from .gates import GateError, load_exact_processor


CACHE_CONTRACT = "research_gemma4_exact_cache_v1"
_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_SINGLE_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_weight_manifest(snapshot: Path) -> dict[str, Any]:
    """Prove that every model-weight shard named by the snapshot is present."""

    indices = [snapshot / name for name in _WEIGHT_INDEX_NAMES if (snapshot / name).is_file()]
    if indices:
        required: set[str] = set()
        for index in indices:
            try:
                payload = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise GateError(f"Invalid model weight index {index}: {exc}") from exc
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise GateError(f"Model weight index has no non-empty weight_map: {index}")
            for filename in weight_map.values():
                relative = Path(str(filename))
                if relative.is_absolute() or ".." in relative.parts:
                    raise GateError(
                        f"Model weight index contains an unsafe shard path: {filename!r}"
                    )
                required.add(relative.as_posix())
        missing = []
        empty = []
        logical_bytes = 0
        for filename in sorted(required):
            shard = snapshot / filename
            if not shard.is_file():
                missing.append(filename)
                continue
            size = shard.stat().st_size
            if size <= 0:
                empty.append(filename)
            logical_bytes += size
        if missing or empty:
            raise GateError(
                "Cached model weight manifest is incomplete: "
                f"missing={missing}, empty={empty}"
            )
        return {
            "mode": "sharded_index",
            "indices": [path.name for path in indices],
            "weight_files": len(required),
            "weight_bytes": logical_bytes,
        }

    singles = [
        snapshot / name
        for name in _SINGLE_WEIGHT_NAMES
        if (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
    ]
    if not singles:
        raise GateError(
            "Cached snapshot has neither a complete weight index nor a non-empty "
            f"single model weight file ({', '.join(_SINGLE_WEIGHT_NAMES)})"
        )
    return {
        "mode": "single_file",
        "indices": [],
        "weight_files": len(singles),
        "weight_bytes": sum(path.stat().st_size for path in singles),
    }


def cache_model(choice: str, *, allow_download: bool) -> dict[str, Any]:
    spec = get_research_model_spec(choice)
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
    except Exception as exc:  # pragma: no cover - live dependency.
        raise GateError(f"Research cache dependencies are unavailable: {exc}") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_files_only=not allow_download,
        )
    ).resolve(strict=True)
    processor, loader = load_exact_processor(
        spec.model_id,
        spec.revision,
        local_files_only=True,
        trust_remote_code=spec.trust_remote_code,
    )
    del processor
    if loader != "AutoProcessor":
        raise GateError(f"{choice}: cached model did not load through AutoProcessor")
    config = AutoConfig.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        local_files_only=True,
        trust_remote_code=spec.trust_remote_code,
    )
    observed_architecture = str(getattr(config, "model_type", ""))
    if observed_architecture != spec.architecture:
        raise GateError(
            f"{choice}: cached architecture {observed_architecture!r} does not match "
            f"{spec.architecture!r}"
        )
    weight_manifest = _verify_weight_manifest(snapshot)
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "architecture": spec.architecture,
        "prompt_profile": spec.prompt_profile,
        "snapshot": str(snapshot),
        "files": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
        "weight_manifest": weight_manifest,
        "processor_loader": loader,
        "passed": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Pre-cache pinned E4B/12B snapshots")
    result.add_argument("--model-choice", action="append", choices=("e4b", "12b"))
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    choices = args.model_choice or [GEMMA4_E4B.key, GEMMA4_12B.key]
    try:
        models = [
            cache_model(choice, allow_download=args.allow_download) for choice in choices
        ]
        report = {
            "contract": CACHE_CONTRACT,
            "passed": all(model["passed"] for model in models),
            "models": models,
        }
        _write_json(args.output.expanduser().resolve(), report)
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

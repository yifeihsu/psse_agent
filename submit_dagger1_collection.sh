#!/bin/bash
#SBATCH --job-name=dagger1_strict
#SBATCH --output=/scratch/yx3882/psse_agent/artifacts/logs/dagger1_%j.out
#SBATCH --error=/scratch/yx3882/psse_agent/artifacts/logs/dagger1_%j.err
#SBATCH --account=torch_pr_627_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|h100|rtx6000"
# NYU's general-account route to RTX Pro 6000 is preemptible and is enabled by
# this exact scheduler comment.  A requeue gets a new immutable attempt
# directory through SLURM_RESTART_COUNT; canonical collector outputs still
# refuse overwrite if a prior attempt reached publication.
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=yx3882@nyu.edu

# Canonical Slurm payload for one DAgger-1 collection.  Torch assigns every
# accessible partition that matches the resource request, so this script does
# not pin a partition or QOS.  Command-line sbatch options may narrow the
# constraint, for example:
#
#   sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=128G \
#     --export=ALL,EXPECTED_ACCELERATOR_CLASS=rtx6000,... \
#     submit_dagger1_collection.sh
#
# ``rtx6000`` is NYU's Slurm spelling for RTX Pro 6000.  Runtime attestation
# additionally requires the NVIDIA RTX Pro name and at least 90,000 MiB.
# The four-CPU override avoids leaving otherwise-free RTX Pro GPUs stranded by
# CPU packing on the eight-GPU nodes; keep the 128-GiB host-memory request.
# EXPECTED_ACCELERATOR_CLASS=auto accepts any approved release accelerator.
# A class-specific canary must instead use h100, h200, or rtx6000 so runtime
# evidence cannot be attributed to the wrong accelerator class.
#
# The canonical collection wrapper enables --overlap-policy-audit, which
# overlaps CPU-side private-policy auditing with model inference without
# changing the ordered collection result.  The immutable run receipt reports
# mean/nonzero/p95 GPU use, power, and peak memory from the 1-second telemetry.
# NYU's communicated 75% warning and 60% cancellation-risk thresholds are
# classifications only: low utilization does not invalidate an otherwise
# scientifically complete run.  Classification requires at least 60 valid,
# strictly ordered 1-second samples spanning 59 seconds with no gap above three
# seconds, bound to the release-attested device, and covering the measured
# collection interval; shorter evidence is labeled insufficient rather than
# extrapolated.
# Collection success has a separate one-valid-sample evidence floor.  Missing
# telemetry still fails closed, while malformed-row counts remain explicit.
# Collector and telemetry commands run in isolated sessions so TERM/KILL can
# quiesce every descendant before artifacts are hashed into the receipt.

set -euo pipefail
umask 077

PY=${PY:-/scratch/yx3882/.conda/envs/unsloth_sft/bin/python}
REPO=${REPO:-/scratch/yx3882/psse_agent}
COMMIT=${COMMIT:-}
D0_DIR=${D0_DIR:-}
D1_DIR=${D1_DIR:-}
LEARNER=${LEARNER:-}
REV=${REV:-}
HFROOT=${HFROOT:-/scratch/yx3882/.cache/huggingface}
MODE=${MODE:-strict}
BETA=${BETA:-0.25}
EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}

if [[ ! "$COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: COMMIT must be the reviewed 40-hex source commit." >&2
    exit 2
fi
if [[ ! "$REV" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: REV must be the exact 64-hex learner adapter tree digest." >&2
    exit 2
fi
COMMIT=${COMMIT,,}
REV=${REV,,}
for variable_name in REPO D0_DIR D1_DIR LEARNER HFROOT; do
    value=${!variable_name}
    if [[ -z "$value" || "$value" != /* ]]; then
        echo "ERROR: $variable_name must be a non-empty absolute path." >&2
        exit 2
    fi
done
case "$MODE" in
    strict|analysis) ;;
    *) echo "ERROR: MODE must be strict or analysis." >&2; exit 2 ;;
esac
if [[ "$BETA" != "0.25" ]]; then
    echo "ERROR: the canonical DAgger-1 launcher requires BETA=0.25." >&2
    exit 2
fi
case "$EXPECTED_ACCELERATOR_CLASS" in
    auto|h100|h200|rtx6000) ;;
    *)
        echo "ERROR: EXPECTED_ACCELERATOR_CLASS must be auto, h100, h200, or rtx6000." >&2
        exit 2
        ;;
esac
for path in "$PY" "$REPO" "$D0_DIR" "$D1_DIR" "$LEARNER"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required collection input is missing: $path" >&2
        exit 2
    fi
done
if [[ ! -x "$PY" ]]; then
    echo "ERROR: PY is not executable: $PY" >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1 || ! command -v ps >/dev/null 2>&1; then
    echo "ERROR: setsid and ps are required for bounded process-group cleanup." >&2
    exit 2
fi
cd "$REPO"

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
SLURM_JOB_ID=${SLURM_JOB_ID:-}
if [[ ! "$SLURM_JOB_ID" =~ ^[1-9][0-9]{0,19}$ ]]; then
    echo "ERROR: SLURM_JOB_ID must be a canonical positive decimal job ID." >&2
    exit 2
fi
if [[ ! "$RESTART_COUNT" =~ ^(0|[1-9][0-9]{0,19})$ ]]; then
    echo "ERROR: SLURM_RESTART_COUNT must be a canonical non-negative integer." >&2
    exit 2
fi
RECEIPT_ROOT="$D1_DIR/run_receipts"
RECEIPT_JOB_DIR="$RECEIPT_ROOT/$SLURM_JOB_ID"
RECEIPT_DIR="$RECEIPT_JOB_DIR/attempt-$RESTART_COUNT"
OUTPUT_LOCK_ROOT="$D1_DIR/.dagger1_collection_owners"
OUTPUT_LOCK_DIR="$OUTPUT_LOCK_ROOT/$MODE"
RECEIPT_READY=0
OUTPUT_LOCK_OWNED=0
ARTIFACT_OWNERSHIP_ESTABLISHED=0
COLLECT_PID=""
COLLECT_PGID=""
COLLECT_SUPERVISOR_PID=""
SUPERVISOR_TRANSPORT_RC=""
COLLECT_RC=""
COLLECTION_STARTED=0
COLLECTION_COMPLETED=0
COLLECTION_PROCESS_STATUS="not_started"
COLLECTION_PROCESS_EXIT_CODE=""
COLLECTION_PROCESS_QUIESCED=1
COLLECTION_KILL_ESCALATED=0
COLLECTION_CLEANUP_ACTIVE=0
TERMINATION_SIGNAL=""
NVML_PID=""
NVML_PGID=""
NVML_EXIT_CODE=""
NVML_STATUS="not_started"
NVML_PROCESS_QUIESCED=1
NVML_KILL_ESCALATED=0
TELEMETRY="$RECEIPT_DIR/nvml_1s.csv"
NVML_DEVICE_ATTESTATION="$RECEIPT_DIR/nvml_device_attestation.csv"
FINAL_EXIT_CODE=0
FINALIZING=0
RECEIPT_TERMINAL_STATE_FROZEN=0
RECEIPT_PUBLISHED=0
RECEIPT_PUBLICATION_INDETERMINATE=0
POST_FREEZE_SIGNAL_NAME=""
POST_FREEZE_SIGNAL_EXIT_CODE=""
COLLECTION_INTERVAL_START_EPOCH=""
COLLECTION_INTERVAL_END_EPOCH=""
COLLECTION_TREE_QUIESCED_MARKER="$RECEIPT_DIR/collection_tree_quiesced"
COLLECTION_SUPERVISOR_STATUS="$RECEIPT_DIR/collection_supervisor_status.json"
SUPERVISOR_TRANSPORT_SUCCESS_RC=86
SUPERVISOR_TRANSPORT_FAILURE_EXIT_CODE=48

COLLECT_TERM_GRACE_TICKS=50
COLLECT_KILL_GRACE_TICKS=20
NVML_TERM_GRACE_TICKS=10
NVML_KILL_GRACE_TICKS=10

mark_collection_interval_end() {
    if [[ "$COLLECTION_STARTED" -eq 1 \
        && -z "$COLLECTION_INTERVAL_END_EPOCH" ]]; then
        COLLECTION_INTERVAL_END_EPOCH=$(date +%s.%N)
    fi
}

release_output_lock() {
    if [[ "${OUTPUT_LOCK_OWNED:-0}" -ne 1 ]]; then
        return
    fi
    if [[ "$RECEIPT_PUBLICATION_INDETERMINATE" -eq 1 ]]; then
        echo "ERROR: retaining the D1 mode lock because receipt rollback durability is indeterminate." >&2
        return
    fi
    if [[ "$COLLECTION_STARTED" -eq 1 \
        && "$COLLECTION_PROCESS_QUIESCED" -ne 1 ]]; then
        echo "ERROR: retaining the D1 mode lock because collection-tree quiescence is unverified." >&2
        return
    fi
    if ! rmdir -- "$OUTPUT_LOCK_DIR"; then
        echo "ERROR: failed to release the owned D1 mode lock: $OUTPUT_LOCK_DIR" >&2
        # Once the immutable receipt is published its terminal status is the
        # exit-code authority.  Preserve the lock as fail-closed evidence, but
        # never create a receipt/process contradiction by changing only the
        # shell status after that point.
        if [[ "$RECEIPT_PUBLISHED" -ne 1 && "$FINAL_EXIT_CODE" -eq 0 ]]; then
            FINAL_EXIT_CODE=46
        fi
        return
    fi
    OUTPUT_LOCK_OWNED=0
}

apply_collection_supervisor_status() {
    local parsed_status
    local parser_rc
    local reported_collection_rc
    local wrapper_completed
    local kill_escalated
    local supervisor_signal

    if [[ "$SUPERVISOR_TRANSPORT_RC" != "$SUPERVISOR_TRANSPORT_SUCCESS_RC" ]]; then
        COLLECTION_PROCESS_QUIESCED=0
        COLLECTION_PROCESS_STATUS="supervisor_transport_failure"
        COLLECT_RC=$SUPERVISOR_TRANSPORT_FAILURE_EXIT_CODE
        return
    fi
    set +e
    parsed_status=$("$PY" - "$COLLECTION_SUPERVISOR_STATUS" \
        "$COLLECTION_TREE_QUIESCED_MARKER" "$COLLECT_SUPERVISOR_PID" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


status_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
expected_pid = int(sys.argv[3])
status_bytes = status_path.read_bytes()
expected_marker = hashlib.sha256(status_bytes).hexdigest() + "\n"
if marker_path.read_text(encoding="ascii") != expected_marker:
    raise SystemExit("supervisor status digest marker mismatch")
payload = json.loads(status_bytes)
if payload.get("contract") != "dagger1_collection_subreaper_v2":
    raise SystemExit("unexpected supervisor status contract")
if payload.get("supervisor_pid") != expected_pid:
    raise SystemExit("supervisor PID binding mismatch")
collection_rc = payload.get("collection_exit_code")
wrapper_completed = payload.get("wrapper_completed_before_signal")
kill_escalated = payload.get("kill_escalated")
tree_quiesced = payload.get("descendant_tree_quiesced")
termination_signal = payload.get("termination_signal")
if type(collection_rc) is not int or not 0 <= collection_rc <= 255:
    raise SystemExit("invalid reported collection exit code")
if type(wrapper_completed) is not bool or type(kill_escalated) is not bool:
    raise SystemExit("invalid supervisor boolean evidence")
if tree_quiesced is not True:
    raise SystemExit("supervisor did not attest tree quiescence")
if termination_signal not in {None, "SIGINT", "SIGTERM"}:
    raise SystemExit("invalid supervisor termination signal")
print(
    "|".join(
        (
            str(collection_rc),
            "1" if wrapper_completed else "0",
            "1" if kill_escalated else "0",
            termination_signal.removeprefix("SIG")
            if termination_signal is not None
            else "",
        )
    )
)
PY
    )
    parser_rc=$?
    if [[ "$parser_rc" -ne 0 ]]; then
        COLLECTION_PROCESS_QUIESCED=0
        COLLECTION_PROCESS_STATUS="invalid_supervisor_transport_evidence"
        COLLECT_RC=$SUPERVISOR_TRANSPORT_FAILURE_EXIT_CODE
        return
    fi
    IFS='|' read -r reported_collection_rc wrapper_completed \
        kill_escalated supervisor_signal <<< "$parsed_status"
    COLLECT_RC=$reported_collection_rc
    COLLECTION_PROCESS_EXIT_CODE=$reported_collection_rc
    COLLECTION_COMPLETED=$wrapper_completed
    COLLECTION_PROCESS_QUIESCED=1
    if [[ "$kill_escalated" == "1" ]]; then
        COLLECTION_KILL_ESCALATED=1
        COLLECTION_PROCESS_STATUS="killed_after_timeout"
        COLLECTION_PROCESS_EXIT_CODE=137
    fi
    if [[ -n "$supervisor_signal" && -z "$TERMINATION_SIGNAL" ]]; then
        TERMINATION_SIGNAL=$supervisor_signal
    fi
}

process_group_has_live_members() {
    local pgid=$1
    local observed_pgid
    local process_table
    local state
    # Use only POSIX-style ps columns.  Some supported login environments do
    # not provide procps' GNU --pgroup selector.
    if ! process_table=$(ps -e -o pgid= -o stat= 2>/dev/null); then
        # A failed probe must never authorize an unbounded wait or artifact
        # hashing.  Conservatively report the group as live.
        return 0
    fi
    while read -r observed_pgid state; do
        if [[ "$observed_pgid" == "$pgid" && -n "$state" \
            && "${state:0:1}" != "Z" ]]; then
            return 0
        fi
    done <<< "$process_table"
    return 1
}

wait_for_process_group() {
    local pgid=$1
    local ticks=$2
    local tick
    for ((tick = 0; tick < ticks; tick++)); do
        if ! process_group_has_live_members "$pgid"; then
            return 0
        fi
        sleep 0.1
    done
    ! process_group_has_live_members "$pgid"
}

reap_process_leader() {
    local pid=$1
    local result_variable=$2
    local leader_rc
    if [[ -z "$pid" ]]; then
        return
    fi
    # Call only after the isolated process group is quiescent, so wait cannot
    # block finalization.  It merely reaps the direct session leader.
    set +e
    wait "$pid" 2>/dev/null
    leader_rc=$?
    printf -v "$result_variable" '%s' "$leader_rc"
}

terminate_collection_group() {
    local signal_name=${1:-TERM}
    local pgid=${COLLECT_PGID:-}
    local leader=${COLLECT_PID:-}

    if [[ -z "$pgid" ]]; then
        COLLECTION_PROCESS_QUIESCED=1
        return
    fi
    COLLECTION_CLEANUP_ACTIVE=1
    COLLECTION_PROCESS_STATUS="termination_requested"
    if process_group_has_live_members "$pgid"; then
        kill -"$signal_name" -- "-$pgid" 2>/dev/null || true
    fi
    if wait_for_process_group "$pgid" "$COLLECT_TERM_GRACE_TICKS"; then
        COLLECTION_PROCESS_STATUS="terminated_by_signal"
        COLLECTION_PROCESS_QUIESCED=1
    else
        COLLECTION_KILL_ESCALATED=1
        kill -KILL -- "-$pgid" 2>/dev/null || true
        if wait_for_process_group "$pgid" "$COLLECT_KILL_GRACE_TICKS"; then
            COLLECTION_PROCESS_STATUS="killed_after_timeout"
            COLLECTION_PROCESS_QUIESCED=1
        else
            COLLECTION_PROCESS_STATUS="kill_timeout"
            COLLECTION_PROCESS_QUIESCED=0
        fi
    fi
    if [[ "$COLLECTION_PROCESS_QUIESCED" -eq 1 ]]; then
        reap_process_leader "$leader" COLLECTION_PROCESS_EXIT_CODE
        SUPERVISOR_TRANSPORT_RC=$COLLECTION_PROCESS_EXIT_CODE
        COLLECT_PID=""
        COLLECT_PGID=""
        apply_collection_supervisor_status
    fi
    COLLECTION_CLEANUP_ACTIVE=0
}

stop_nvml() {
    local pgid=${NVML_PGID:-}
    local leader=${NVML_PID:-}

    if [[ -z "$pgid" ]]; then
        NVML_STATUS="not_started"
        NVML_EXIT_CODE=""
        NVML_PROCESS_QUIESCED=1
        return
    fi
    if ! process_group_has_live_members "$pgid"; then
        NVML_STATUS="exited_before_finalize"
        NVML_PROCESS_QUIESCED=1
        reap_process_leader "$leader" NVML_EXIT_CODE
        NVML_PID=""
        NVML_PGID=""
        return
    fi

    kill -TERM -- "-$pgid" 2>/dev/null || true
    if wait_for_process_group "$pgid" "$NVML_TERM_GRACE_TICKS"; then
        NVML_STATUS="terminated_by_launcher"
        NVML_PROCESS_QUIESCED=1
    else
        NVML_KILL_ESCALATED=1
        kill -KILL -- "-$pgid" 2>/dev/null || true
        if wait_for_process_group "$pgid" "$NVML_KILL_GRACE_TICKS"; then
            NVML_STATUS="killed_after_timeout"
            NVML_PROCESS_QUIESCED=1
        else
            NVML_STATUS="kill_timeout"
            NVML_PROCESS_QUIESCED=0
        fi
    fi
    if [[ "$NVML_PROCESS_QUIESCED" -eq 1 ]]; then
        reap_process_leader "$leader" NVML_EXIT_CODE
        NVML_PID=""
        NVML_PGID=""
    fi
}

publish_receipt() {
    local proposed_exit_code=$1
    local publisher_output
    local publisher_rc

    FINAL_EXIT_CODE=$proposed_exit_code
    if [[ "${RECEIPT_READY:-0}" -ne 1 ]]; then
        return
    fi
    stop_nvml
    # TERM/INT remain non-reentrant while the monitor is stopped and the
    # immutable receipt is linked and directory-fsynced.  A signal delivered
    # during monitor cleanup is therefore folded into this one publication.
    if [[ -n "$DEFERRED_SIGNAL_NAME" ]]; then
        TERMINATION_SIGNAL=$DEFERRED_SIGNAL_NAME
        proposed_exit_code=$DEFERRED_SIGNAL_EXIT_CODE
        FINAL_EXIT_CODE=$proposed_exit_code
    fi
    # This is the terminal-state linearization point.  Signals observed after
    # it remain non-reentrant through link+fsync, but cannot change only the
    # shell's exit code after immutable terminal fields have been serialized.
    RECEIPT_TERMINAL_STATE_FROZEN=1
    export PY REPO COMMIT D0_DIR D1_DIR LEARNER REV HFROOT MODE BETA
    export SLURM_JOB_ID COLLECT_RC RECEIPT_DIR EXPECTED_ACCELERATOR_CLASS TELEMETRY
    export NVML_DEVICE_ATTESTATION
    export RESTART_COUNT COLLECTION_STARTED COLLECTION_COMPLETED
    export COLLECTION_PROCESS_STATUS COLLECTION_PROCESS_EXIT_CODE
    export COLLECTION_PROCESS_QUIESCED COLLECTION_KILL_ESCALATED
    export SUPERVISOR_TRANSPORT_RC
    export COLLECTION_INTERVAL_START_EPOCH COLLECTION_INTERVAL_END_EPOCH
    export ARTIFACT_OWNERSHIP_ESTABLISHED
    export TERMINATION_SIGNAL NVML_EXIT_CODE NVML_STATUS
    export NVML_PROCESS_QUIESCED NVML_KILL_ESCALATED
    export PROPOSED_EXIT_CODE="$proposed_exit_code"
    export LAUNCHER_PID=$$
    set +e
    publisher_output=$("$PY" - <<'PY'
from __future__ import annotations

import csv
from datetime import datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import time
from typing import Any


TELEMETRY_FAILURE_EXIT_CODE = 40
NYU_WARNING_THRESHOLD_PERCENT = 75.0
NYU_CANCELLATION_RISK_THRESHOLD_PERCENT = 60.0
UTILIZATION_CLASSIFICATION_MINIMUM_SAMPLES = 60
UTILIZATION_CLASSIFICATION_MINIMUM_SPAN_SECONDS = 59.0
UTILIZATION_CLASSIFICATION_MAX_GAP_SECONDS = 3.0
COLLECTION_INTERVAL_COVERAGE_TOLERANCE_SECONDS = 3.0
ATTESTATION_MEMORY_TOLERANCE_MIB = 16.0
TELEMETRY_MEMORY_IDENTITY_TOLERANCE_MIB = 0.5


class ReceiptPublisherSignal(RuntimeError):
    """Convert publisher termination into the owned-link rollback path."""


def raise_publisher_signal(signum: int, _frame: object) -> None:
    raise ReceiptPublisherSignal(signal.Signals(signum).name)


signal.signal(signal.SIGINT, raise_publisher_signal)
signal.signal(signal.SIGTERM, raise_publisher_signal)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    return int(value) if value else None


def release_device_identity(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        devices = payload["devices"]
        if payload["device_count"] != 1 or len(devices) != 1:
            return None
        device = devices[0]
        name = " ".join(str(device["name"]).upper().split())
        total_memory_bytes = int(device["total_memory_bytes"])
        if device["index"] != 0 or not name or total_memory_bytes <= 0:
            return None
        return {
            "name": name,
            "total_memory_mib": total_memory_bytes / (1024**2),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def nvml_device_identity(
    path: Path,
    *,
    release_device: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if release_device is None:
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) != 1 or len(rows[0]) != 3:
            return None
        uuid, raw_name, raw_memory = (value.strip() for value in rows[0])
        name = " ".join(raw_name.upper().split())
        memory_total = float(raw_memory)
        if not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9-]+", uuid):
            return None
        if name != release_device["name"]:
            return None
        if (
            abs(memory_total - release_device["total_memory_mib"])
            > ATTESTATION_MEMORY_TOLERANCE_MIB
        ):
            return None
        return {"uuid": uuid, "name": name, "total_memory_mib": memory_total}
    except (OSError, UnicodeError, ValueError, csv.Error):
        return None


def telemetry_audit(
    path: Path,
    attestation_path: Path,
    nvml_attestation_path: Path,
) -> dict[str, Any]:
    release_device = release_device_identity(attestation_path)
    attested_device = nvml_device_identity(
        nvml_attestation_path,
        release_device=release_device,
    )
    empty = {
        "header_valid": False,
        "total_data_rows": 0,
        "invalid_data_rows": 0,
        "identity_mismatch_rows": 0,
        "release_attestation_device_valid": release_device is not None,
        "nvml_device_attestation_valid": attested_device is not None,
        "device_identity_match": False,
        "telemetry_device_uuid": None,
        "telemetry_device_name": None,
        "telemetry_device_total_memory_mib": None,
        "samples": [],
    }
    if not path.is_file():
        return empty
    expected_header = (
        "timestamp",
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.sm",
    )

    def normalized_header(value: str) -> str:
        return value.split("[", 1)[0].strip().lower()

    def number(value: str) -> float | None:
        match = re.fullmatch(
            r"[ ]*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?:[ ]+[^,]+)?[ ]*",
            value,
        )
        return float(match.group(1)) if match else None

    def sample_timestamp(value: str) -> float | None:
        text = value.strip()
        for timestamp_format in (
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, timestamp_format)
            except ValueError:
                continue
            # nvidia-smi emits host-local wall time without an offset.  The
            # launcher interval comes from epoch time on the same host, so use
            # the host-local timezone conversion rather than treating the text
            # as UTC.
            return parsed.timestamp()
        return None

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return empty
            normalized = tuple(map(normalized_header, header))
            if normalized[:-1] != expected_header[:-1] or normalized[-1] not in {
                "clocks.sm",
                "clocks.current.sm",
            }:
                remaining_rows = sum(1 for _ in reader)
                return {
                    **empty,
                    "header_valid": False,
                    "total_data_rows": remaining_rows,
                    "invalid_data_rows": remaining_rows,
                }
            valid: list[
                tuple[float, str, str, float, float, float, float]
            ] = []
            total_rows = 0
            invalid_rows = 0
            identity_mismatch_rows = 0
            observed_uuid: str | None = None
            observed_name: str | None = None
            observed_total_memory: float | None = None
            for row in reader:
                total_rows += 1
                if len(row) != len(expected_header):
                    invalid_rows += 1
                    continue
                timestamp = sample_timestamp(row[0])
                utilization = number(row[4])
                memory_used = number(row[5])
                memory_total = number(row[6])
                power = number(row[7])
                clocks = number(row[8])
                uuid = row[2].strip()
                name = " ".join(row[3].upper().split())
                structurally_valid = bool(
                    timestamp is not None
                    and row[1].strip().isdigit()
                    and re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9-]+", uuid)
                    and name
                    and utilization is not None
                    and 0.0 <= utilization <= 100.0
                    and memory_used is not None
                    and memory_used >= 0.0
                    and memory_total is not None
                    and memory_total > 0.0
                    and memory_used <= memory_total
                    and power is not None
                    and power >= 0.0
                    and clocks is not None
                    and clocks >= 0.0
                )
                if not structurally_valid:
                    invalid_rows += 1
                    continue
                if observed_uuid is None:
                    observed_uuid = uuid
                    observed_name = name
                    observed_total_memory = memory_total
                identity_matches = bool(
                    attested_device is not None
                    and uuid == attested_device["uuid"]
                    and name == attested_device["name"]
                    and abs(
                        memory_total - attested_device["total_memory_mib"]
                    ) <= TELEMETRY_MEMORY_IDENTITY_TOLERANCE_MIB
                )
                if not identity_matches:
                    invalid_rows += 1
                    identity_mismatch_rows += 1
                    continue
                valid.append(
                    (
                        timestamp,
                        uuid,
                        name,
                        memory_total,
                        utilization,
                        memory_used,
                        power,
                    )
                )
            return {
                "header_valid": True,
                "total_data_rows": total_rows,
                "invalid_data_rows": invalid_rows,
                "identity_mismatch_rows": identity_mismatch_rows,
                "release_attestation_device_valid": release_device is not None,
                "nvml_device_attestation_valid": attested_device is not None,
                "device_identity_match": bool(valid and identity_mismatch_rows == 0),
                "telemetry_device_uuid": observed_uuid,
                "telemetry_device_name": observed_name,
                "telemetry_device_total_memory_mib": observed_total_memory,
                "samples": valid,
            }
    except (OSError, UnicodeError, csv.Error):
        return empty


def telemetry_summary(
    audit: dict[str, Any],
    *,
    collection_start: float | None,
    collection_end: float | None,
) -> dict[str, Any]:
    samples = audit["samples"]
    base = {
        "header_valid": audit["header_valid"],
        "total_data_rows": audit["total_data_rows"],
        "invalid_data_rows": audit["invalid_data_rows"],
        "identity_mismatch_rows": audit["identity_mismatch_rows"],
        "valid_data_rows": len(samples),
        "release_attestation_device_valid": audit[
            "release_attestation_device_valid"
        ],
        "nvml_device_attestation_valid": audit[
            "nvml_device_attestation_valid"
        ],
        "device_identity_match": audit["device_identity_match"],
        "telemetry_device_uuid": audit["telemetry_device_uuid"],
        "telemetry_device_name": audit["telemetry_device_name"],
        "telemetry_device_total_memory_mib": audit[
            "telemetry_device_total_memory_mib"
        ],
        "timestamp_span_seconds": None,
        "max_timestamp_gap_seconds": None,
        "timestamps_strictly_increasing": False,
        "collection_interval_duration_seconds": None,
        "first_sample_start_offset_seconds": None,
        "last_sample_end_offset_seconds": None,
        "collection_interval_coverage": False,
        "representative_coverage": False,
    }
    if not samples:
        return {
            **base,
            "mean_gpu_utilization_percent": None,
            "nonzero_gpu_duty_percent": None,
            "p95_gpu_utilization_percent": None,
            "mean_power_draw_watts": None,
            "max_memory_used_mib": None,
        }
    timestamps = [sample[0] for sample in samples]
    utilizations = [sample[4] for sample in samples]
    memory_used = [sample[5] for sample in samples]
    power_draw = [sample[6] for sample in samples]
    gaps = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ]
    timestamps_increasing = all(gap > 0.0 for gap in gaps)
    timestamp_span = timestamps[-1] - timestamps[0]
    max_gap = max(gaps) if gaps else None
    interval_available = bool(
        collection_start is not None
        and collection_end is not None
        and collection_end >= collection_start
    )
    interval_duration = (
        collection_end - collection_start if interval_available else None
    )
    first_start_offset = (
        timestamps[0] - collection_start if interval_available else None
    )
    last_end_offset = (
        timestamps[-1] - collection_end if interval_available else None
    )
    interval_coverage = bool(
        interval_available
        and abs(first_start_offset) <= COLLECTION_INTERVAL_COVERAGE_TOLERANCE_SECONDS
        and abs(last_end_offset) <= COLLECTION_INTERVAL_COVERAGE_TOLERANCE_SECONDS
    )
    representative = bool(
        audit["header_valid"]
        and audit["invalid_data_rows"] == 0
        and audit["device_identity_match"]
        and len(samples) >= UTILIZATION_CLASSIFICATION_MINIMUM_SAMPLES
        and timestamps_increasing
        and timestamp_span >= UTILIZATION_CLASSIFICATION_MINIMUM_SPAN_SECONDS
        and max_gap is not None
        and max_gap <= UTILIZATION_CLASSIFICATION_MAX_GAP_SECONDS
        and interval_coverage
    )
    ordered_utilizations = sorted(utilizations)
    p95_index = math.ceil(0.95 * len(ordered_utilizations)) - 1
    return {
        **base,
        "timestamp_span_seconds": round(timestamp_span, 6),
        "max_timestamp_gap_seconds": (
            round(max_gap, 6) if max_gap is not None else None
        ),
        "timestamps_strictly_increasing": timestamps_increasing,
        "collection_interval_duration_seconds": (
            round(interval_duration, 6) if interval_duration is not None else None
        ),
        "first_sample_start_offset_seconds": (
            round(first_start_offset, 6)
            if first_start_offset is not None
            else None
        ),
        "last_sample_end_offset_seconds": (
            round(last_end_offset, 6) if last_end_offset is not None else None
        ),
        "collection_interval_coverage": interval_coverage,
        "representative_coverage": representative,
        "mean_gpu_utilization_percent": round(
            sum(utilizations) / len(utilizations), 6
        ),
        "nonzero_gpu_duty_percent": round(
            100.0 * sum(value > 0.0 for value in utilizations) / len(utilizations),
            6,
        ),
        # Nearest-rank p95 is an observed sample and is deterministic for
        # short canaries as well as full collection telemetry.
        "p95_gpu_utilization_percent": ordered_utilizations[p95_index],
        "mean_power_draw_watts": round(sum(power_draw) / len(power_draw), 6),
        "max_memory_used_mib": max(memory_used),
    }


def nyu_utilization_classification(
    mean_utilization: float | None, *, representative: bool
) -> str:
    if mean_utilization is None:
        return "unavailable"
    if not representative:
        return "insufficient_representative_coverage"
    if mean_utilization < NYU_CANCELLATION_RISK_THRESHOLD_PERCENT:
        return "cancellation_risk_below_60_percent"
    if mean_utilization < NYU_WARNING_THRESHOLD_PERCENT:
        return "warning_below_75_percent"
    return "meets_or_exceeds_75_percent"


receipt_dir = Path(os.environ["RECEIPT_DIR"])
d1_dir = Path(os.environ["D1_DIR"])
telemetry = Path(os.environ["TELEMETRY"])
mode = os.environ["MODE"]
prefix = "training_beta025" if mode == "strict" else "analysis_beta025"
artifact_names = (
    f"{prefix}.jsonl",
    f"{prefix}.all.jsonl",
    f"{prefix}.jsonl.manifest.json",
    f"{prefix}.failed-collection/failure_evidence.json",
)
collection_process_quiesced = os.environ["COLLECTION_PROCESS_QUIESCED"] == "1"
supervisor_transport_exit_code = optional_int("SUPERVISOR_TRANSPORT_RC")
supervisor_transport_authenticated = bool(
    supervisor_transport_exit_code == 86 and collection_process_quiesced
)
artifact_hashing_skipped = (
    os.environ["ARTIFACT_OWNERSHIP_ESTABLISHED"] != "1"
    or not collection_process_quiesced
)
artifacts = {
    name: None if artifact_hashing_skipped else sha256(d1_dir / name)
    for name in artifact_names
}
nvml_process_quiesced = os.environ["NVML_PROCESS_QUIESCED"] == "1"
# Never read or hash a file that a timed-out monitor could still mutate.  The
# immutable receipt preserves the cleanup failure, but does not bless a racy
# telemetry snapshot as evidence.
telemetry_hashing_skipped = not nvml_process_quiesced
audit = (
    telemetry_audit(
        telemetry,
        receipt_dir / "accelerator_attestation.json",
        Path(os.environ["NVML_DEVICE_ATTESTATION"]),
    )
    if nvml_process_quiesced
    else {
        "header_valid": False,
        "total_data_rows": 0,
        "invalid_data_rows": 0,
        "identity_mismatch_rows": 0,
        "release_attestation_device_valid": False,
        "nvml_device_attestation_valid": False,
        "device_identity_match": False,
        "telemetry_device_uuid": None,
        "telemetry_device_name": None,
        "telemetry_device_total_memory_mib": None,
        "samples": [],
    }
)


def optional_float(name: str) -> float | None:
    value = os.environ.get(name, "")
    return float(value) if value else None


summary = telemetry_summary(
    audit,
    collection_start=optional_float("COLLECTION_INTERVAL_START_EPOCH"),
    collection_end=optional_float("COLLECTION_INTERVAL_END_EPOCH"),
)
valid_samples = summary["valid_data_rows"]
utilization_classification = nyu_utilization_classification(
    summary["mean_gpu_utilization_percent"],
    representative=summary["representative_coverage"],
)
monitor_status = os.environ["NVML_STATUS"]
monitor_exit_code = optional_int("NVML_EXIT_CODE")
collection_exit_code = optional_int("COLLECT_RC")
proposed_exit_code = int(os.environ["PROPOSED_EXIT_CODE"])
termination_signal = os.environ.get("TERMINATION_SIGNAL") or None
telemetry_requirement_met = (
    valid_samples >= 1
    and summary["device_identity_match"]
    and summary["collection_interval_coverage"]
    and monitor_status in {"terminated_by_launcher", "killed_after_timeout"}
    and monitor_exit_code in {0, 137, 143}
    and nvml_process_quiesced
)
final_exit_code = proposed_exit_code
if proposed_exit_code == 0 and not collection_process_quiesced:
    final_exit_code = 43
if (
    final_exit_code == 0
    and collection_exit_code == 0
    and not telemetry_requirement_met
):
    final_exit_code = TELEMETRY_FAILURE_EXIT_CODE

payload = {
    "contract": "dagger1_slurm_run_receipt_v3",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_restart_count": int(os.environ["RESTART_COUNT"]),
    "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
    "slurm_job_gres": os.environ.get("SLURM_JOB_GRES"),
    "slurm_constraint": os.environ.get("SLURM_JOB_CONSTRAINTS"),
    "expected_accelerator_class": os.environ["EXPECTED_ACCELERATOR_CLASS"],
    "source_commit": os.environ["COMMIT"].lower(),
    "learner_adapter_tree_sha256": os.environ["REV"].lower(),
    "mode": mode,
    "beta": os.environ["BETA"],
    "collection_started": os.environ["COLLECTION_STARTED"] == "1",
    "collection_completed": os.environ["COLLECTION_COMPLETED"] == "1",
    # This is the canonical wrapper's classified status (0/20/30/other), not
    # the collector subprocess status that the wrapper records in its log.
    "collection_wrapper_exit_code": collection_exit_code,
    "collection_process_status": os.environ["COLLECTION_PROCESS_STATUS"],
    "collection_process_exit_code": optional_int("COLLECTION_PROCESS_EXIT_CODE"),
    "collection_supervisor_transport_exit_code": optional_int(
        "SUPERVISOR_TRANSPORT_RC"
    ),
    "collection_supervisor_transport_success_exit_code": 86,
    "collection_supervisor_transport_authenticated": (
        supervisor_transport_authenticated
    ),
    "collection_process_quiesced": collection_process_quiesced,
    "collection_kill_escalated": os.environ["COLLECTION_KILL_ESCALATED"] == "1",
    "collection_supervisor_status_sha256": sha256(
        receipt_dir / "collection_supervisor_status.json"
    ) if supervisor_transport_authenticated else None,
    "collection_artifact_hashing_skipped": artifact_hashing_skipped,
    "launcher_exit_code": final_exit_code,
    "termination_signal": termination_signal,
    "terminal_state_freeze_contract": (
        "signals_observed_after_terminal_state_freeze_do_not_change_receipted_exit_status"
    ),
    "accelerator_attestation_sha256": sha256(
        receipt_dir / "accelerator_attestation.json"
    ),
    "nvml_device_attestation_sha256": sha256(
        Path(os.environ["NVML_DEVICE_ATTESTATION"])
    ),
    "slurm_job_snapshot_sha256": sha256(receipt_dir / "slurm_job.txt"),
    "nvml_telemetry_sha256": (
        None if telemetry_hashing_skipped else sha256(telemetry)
    ),
    "nvml_telemetry_hashing_skipped": telemetry_hashing_skipped,
    "nvml_monitor_status": monitor_status,
    "nvml_monitor_exit_code": monitor_exit_code,
    "nvml_process_quiesced": nvml_process_quiesced,
    "nvml_kill_escalated": os.environ["NVML_KILL_ESCALATED"] == "1",
    "nvml_valid_sample_count": valid_samples,
    "nvml_success_minimum_valid_samples": 1,
    "nvml_telemetry_requirement_met": telemetry_requirement_met,
    "nvml_telemetry_summary_contract": (
        "attested_device_interval_covered_nvml_rows_nearest_rank_p95_v3"
    ),
    "nvml_utilization_classification_minimum_valid_samples": (
        UTILIZATION_CLASSIFICATION_MINIMUM_SAMPLES
    ),
    "nvml_utilization_classification_minimum_span_seconds": (
        UTILIZATION_CLASSIFICATION_MINIMUM_SPAN_SECONDS
    ),
    "nvml_utilization_classification_max_gap_seconds": (
        UTILIZATION_CLASSIFICATION_MAX_GAP_SECONDS
    ),
    "nvml_collection_interval_coverage_tolerance_seconds": (
        COLLECTION_INTERVAL_COVERAGE_TOLERANCE_SECONDS
    ),
    "nvml_attestation_memory_tolerance_mib": ATTESTATION_MEMORY_TOLERANCE_MIB,
    "nvml_telemetry_memory_identity_tolerance_mib": (
        TELEMETRY_MEMORY_IDENTITY_TOLERANCE_MIB
    ),
    "nvml_telemetry_summary": summary,
    "nyu_gpu_utilization_thresholds_percent": {
        "warning": NYU_WARNING_THRESHOLD_PERCENT,
        "cancellation_risk": NYU_CANCELLATION_RISK_THRESHOLD_PERCENT,
    },
    "nyu_gpu_utilization_classification": utilization_classification,
    "collection_artifact_sha256": artifacts,
}
serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
temporary = receipt_dir / f".run_receipt.{secrets.token_hex(8)}.tmp"
final = receipt_dir / "run_receipt.json"
final_link_created = False
final_link_collision = False
owned_inode: tuple[int, int] | None = None


def unlink_temporary() -> None:
    if os.environ.get("FAIL_RECEIPT_TEMP_UNLINK") == "1":
        raise OSError(errno.EACCES, "injected persistent temporary unlink failure")
    temporary.unlink(missing_ok=True)


try:
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        stat_result = os.fstat(handle.fileno())
        owned_inode = (stat_result.st_dev, stat_result.st_ino)
    os.chmod(temporary, 0o400)
    # A hard link publishes one fully fsynced inode and fails with EEXIST.  It
    # therefore cannot replace evidence even if the destination appears after
    # the exclusive attempt directory was claimed.
    try:
        os.link(temporary, final)
    except FileExistsError:
        final_link_collision = True
        raise
    final_link_created = True
    unlink_temporary()
    directory_fd = os.open(receipt_dir, os.O_RDONLY)
    try:
        if os.environ.get("SIGNAL_BEFORE_RECEIPT_DIR_FSYNC") == "1":
            os.kill(int(os.environ["LAUNCHER_PID"]), signal.SIGTERM)
            # Give the launcher a deterministic opportunity to run its
            # deferred handler while publication is still in the fsync window.
            time.sleep(0.05)
        if os.environ.get("SIGNAL_PUBLISHER_BEFORE_RECEIPT_DIR_FSYNC") == "1":
            os.kill(os.getpid(), signal.SIGTERM)
        if os.environ.get("KILL_PUBLISHER_BEFORE_RECEIPT_DIR_FSYNC") == "1":
            os.kill(os.getpid(), signal.SIGKILL)
        if os.environ.get("FAIL_RECEIPT_DIR_FSYNC") == "1":
            raise OSError(errno.EIO, "injected receipt-directory fsync failure")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException as publication_error:
    temporary_cleanup_error: BaseException | None = None
    final_rollback_error: BaseException | None = None
    try:
        unlink_temporary()
    except BaseException as cleanup_error:
        temporary_cleanup_error = cleanup_error
    if final_link_created:
        try:
            final_stat = os.stat(final, follow_symlinks=False)
            if owned_inode != (final_stat.st_dev, final_stat.st_ino):
                raise RuntimeError("linked receipt inode changed before rollback")
            final.unlink()
            if os.environ.get(
                "KILL_PUBLISHER_AFTER_FINAL_UNLINK_BEFORE_ROLLBACK_FSYNC"
            ) == "1":
                os.kill(os.getpid(), signal.SIGKILL)
            rollback_directory_fd = os.open(receipt_dir, os.O_RDONLY)
            try:
                if os.environ.get("FAIL_RECEIPT_ROLLBACK_FSYNC") == "1":
                    raise OSError(
                        errno.EIO,
                        "injected receipt rollback-directory fsync failure",
                    )
                os.fsync(rollback_directory_fd)
            finally:
                os.close(rollback_directory_fd)
            if final.exists() or final.is_symlink():
                raise RuntimeError("owned receipt link remains after rollback fsync")
        except BaseException as rollback_error:
            final_rollback_error = rollback_error
    if temporary_cleanup_error is not None or final_rollback_error is not None:
        if temporary_cleanup_error is not None:
            print(
                "ERROR: receipt temporary cleanup is indeterminate: "
                f"{temporary_cleanup_error}",
                file=os.sys.stderr,
            )
        if final_rollback_error is not None:
            print(
                "ERROR: receipt publication rollback is indeterminate: "
                f"{final_rollback_error}",
                file=os.sys.stderr,
            )
        raise SystemExit(47) from (
            final_rollback_error or temporary_cleanup_error
        )
    if final_link_collision and not final_link_created:
        # This reserved outcome authenticates a known no-replace collision.
        # The shell can distinguish it from a publisher crash which happened
        # after this invocation linked the final name.
        raise SystemExit(49) from publication_error
    if final_link_created:
        # The owned final was removed and that removal was directory-fsynced.
        # Only this explicit outcome lets the shell classify publication as a
        # known clean rollback; path absence alone is not durability evidence.
        raise SystemExit(50) from publication_error
    raise publication_error
print(final_exit_code)
PY
    )
    publisher_rc=$?
    if [[ "$publisher_rc" -ne 0 ]]; then
        if [[ "$publisher_rc" -eq 49 ]]; then
            echo "ERROR: immutable run receipt publication failed because the final receipt already exists; the pre-existing receipt was preserved." >&2
        elif [[ "$publisher_rc" -eq 50 ]]; then
            echo "ERROR: immutable run receipt publication failed; owned final link rolled back and the removal was directory-fsynced." >&2
        else
            echo "ERROR: immutable run receipt publication is indeterminate; no authenticated clean-rollback outcome was returned." >&2
            FINAL_EXIT_CODE=47
            RECEIPT_PUBLICATION_INDETERMINATE=1
        fi
        if [[ "$RECEIPT_PUBLICATION_INDETERMINATE" -ne 1 \
            && "$proposed_exit_code" -eq 0 ]]; then
            FINAL_EXIT_CODE=42
        fi
        RECEIPT_TERMINAL_STATE_FROZEN=0
        if [[ -n "$POST_FREEZE_SIGNAL_NAME" ]]; then
            DEFERRED_SIGNAL_NAME=$POST_FREEZE_SIGNAL_NAME
            DEFERRED_SIGNAL_EXIT_CODE=$POST_FREEZE_SIGNAL_EXIT_CODE
        fi
        return
    fi
    if [[ ! "$publisher_output" =~ ^(0|[1-9][0-9]{0,2})$ ]] \
        || [[ "$publisher_output" -gt 255 ]]; then
        echo "ERROR: receipt publisher returned an invalid exit code." >&2
        if [[ -e "$RECEIPT_DIR/run_receipt.json" \
            || -L "$RECEIPT_DIR/run_receipt.json" ]]; then
            FINAL_EXIT_CODE=47
            RECEIPT_PUBLICATION_INDETERMINATE=1
        elif [[ "$proposed_exit_code" -eq 0 ]]; then
            FINAL_EXIT_CODE=42
        fi
        RECEIPT_TERMINAL_STATE_FROZEN=0
        if [[ -n "$POST_FREEZE_SIGNAL_NAME" ]]; then
            DEFERRED_SIGNAL_NAME=$POST_FREEZE_SIGNAL_NAME
            DEFERRED_SIGNAL_EXIT_CODE=$POST_FREEZE_SIGNAL_EXIT_CODE
        fi
        return
    fi
    FINAL_EXIT_CODE=$publisher_output
    RECEIPT_PUBLISHED=1
}

on_exit() {
    local proposed_exit_code=$?
    # An EXIT trap reached through errexit inherits the pending -e context.
    # Disable it before any cleanup so a failed setup step cannot short-circuit
    # receipt publication a second time.
    set +e
    trap - EXIT
    FINALIZING=1
    # Do not restore terminating handlers during finalization: a second signal
    # must be recorded, not recurse into cleanup or interrupt no-replace
    # publication.  Keep these handlers through the receipt link and fsync.
    trap 'defer_signal INT 130' INT
    trap 'defer_signal TERM 143' TERM
    if [[ "$COLLECTION_STARTED" -eq 1 \
        && "$COLLECTION_PROCESS_QUIESCED" -ne 1 ]]; then
        if [[ -z "$COLLECT_RC" ]]; then
            COLLECT_RC=$proposed_exit_code
        fi
        terminate_collection_group TERM
        mark_collection_interval_end
    fi
    publish_receipt "$proposed_exit_code"
    if [[ "$RECEIPT_PUBLISHED" -ne 1 && -n "$DEFERRED_SIGNAL_NAME" ]]; then
        TERMINATION_SIGNAL=$DEFERRED_SIGNAL_NAME
        FINAL_EXIT_CODE=$DEFERRED_SIGNAL_EXIT_CODE
    fi
    release_output_lock
    if [[ "${SIGNAL_IMMEDIATELY_BEFORE_LAUNCHER_EXIT:-0}" == "1" ]]; then
        kill -TERM "$$"
        sleep 0.05
    fi
    if [[ "$RECEIPT_PUBLISHED" -eq 1 \
        && -n "$POST_FREEZE_SIGNAL_NAME" ]]; then
        echo "NOTICE: $POST_FREEZE_SIGNAL_NAME arrived after immutable terminal-state freeze; preserving the receipted exit code." >&2
    fi
    # Keep the non-reentrant deferred handlers installed through the exit
    # builtin.  Resetting them here would reopen a final signal-default window
    # after the receipt had made terminal state immutable.
    exit "$FINAL_EXIT_CODE"
}

on_signal() {
    local signal_name=$1
    local signal_exit_code=$2
    if [[ "$FINALIZING" -eq 1 ]]; then
        defer_signal "$signal_name" "$signal_exit_code"
        return
    fi
    if [[ "$COLLECTION_CLEANUP_ACTIVE" -eq 1 ]]; then
        TERMINATION_SIGNAL=$signal_name
        defer_signal "$signal_name" "$signal_exit_code"
        return
    fi
    TERMINATION_SIGNAL=$signal_name
    if [[ "$COLLECTION_STARTED" -eq 1 \
        && "$COLLECTION_PROCESS_QUIESCED" -ne 1 ]]; then
        COLLECT_RC=$signal_exit_code
        # INT is routinely inherited as ignored by asynchronous children.
        # TERM is the portable graceful-stop request for the isolated session;
        # the receipt still records the original signal and exit convention.
        terminate_collection_group TERM
        mark_collection_interval_end
    fi
    exit "$signal_exit_code"
}

DEFERRED_SIGNAL_NAME=""
DEFERRED_SIGNAL_EXIT_CODE=""

defer_signal() {
    # Preserve the first terminating signal as the stable exit convention.
    # Later signals remain deferred and cannot re-enter finalization.
    if [[ "$RECEIPT_TERMINAL_STATE_FROZEN" -eq 1 ]]; then
        if [[ -z "$POST_FREEZE_SIGNAL_NAME" ]]; then
            POST_FREEZE_SIGNAL_NAME=$1
            POST_FREEZE_SIGNAL_EXIT_CODE=$2
        fi
        return
    fi
    if [[ -z "$DEFERRED_SIGNAL_NAME" ]]; then
        DEFERRED_SIGNAL_NAME=$1
        DEFERRED_SIGNAL_EXIT_CODE=$2
    fi
}

install_runtime_signal_traps() {
    trap 'on_signal INT 130' INT
    trap 'on_signal TERM 143' TERM
}

begin_signal_defer() {
    DEFERRED_SIGNAL_NAME=""
    DEFERRED_SIGNAL_EXIT_CODE=""
    trap 'defer_signal INT 130' INT
    trap 'defer_signal TERM 143' TERM
}

end_signal_defer() {
    local signal_name
    local signal_exit_code
    # Restore the terminating handlers before snapshotting deferred state.  A
    # signal before this command is recorded by defer_signal; a signal after
    # it is handled immediately.  Snapshotting first would leave a tiny lost-
    # signal interval between the snapshot and trap restoration.
    install_runtime_signal_traps
    signal_name=$DEFERRED_SIGNAL_NAME
    signal_exit_code=$DEFERRED_SIGNAL_EXIT_CODE
    if [[ -n "$signal_name" ]]; then
        on_signal "$signal_name" "$signal_exit_code"
    fi
}

trap on_exit EXIT
install_runtime_signal_traps

# The finalizers and traps are live before this atomic attempt claim.  Signals
# are deferred only across mkdir and RECEIPT_READY, eliminating the former
# claimed-but-unreceipted interval without allowing attempt reuse.
begin_signal_defer
mkdir -p -- "$RECEIPT_JOB_DIR"
# mkdir without -p is the atomic claim for this attempt leaf.  Do not replace
# this with an existence check followed by mkdir: concurrent launchers could
# both pass the check and then share mutable evidence.
if ! mkdir -- "$RECEIPT_DIR"; then
    install_runtime_signal_traps
    echo "ERROR: refusing to reuse run receipt directory: $RECEIPT_DIR" >&2
    exit 2
fi
RECEIPT_READY=1
end_signal_defer

# A receipt attempt is job-scoped, but canonical D1 outputs are mode-scoped.
# Claim that larger namespace atomically so two distinct Slurm job IDs cannot
# concurrently mutate the same strict (or analysis) products.
begin_signal_defer
mkdir -p -- "$OUTPUT_LOCK_ROOT"
if ! mkdir -- "$OUTPUT_LOCK_DIR"; then
    end_signal_defer
    echo "ERROR: D1 outputs are already owned by another $MODE collection." >&2
    exit 44
fi
OUTPUT_LOCK_OWNED=1
end_signal_defer

HARDWARE_ARGS=()
if [[ "$EXPECTED_ACCELERATOR_CLASS" != "auto" ]]; then
    HARDWARE_ARGS+=(--require-class "$EXPECTED_ACCELERATOR_CLASS")
fi
"$PY" -m psse_env.sft.release_hardware "${HARDWARE_ARGS[@]}" \
    > "$RECEIPT_DIR/accelerator_attestation.json"
nvidia-smi --query-gpu=uuid,name,memory.total \
    --format=csv,noheader,nounits > "$NVML_DEVICE_ATTESTATION"
scontrol show job -o "$SLURM_JOB_ID" > "$RECEIPT_DIR/slurm_job.txt"

if [[ "$MODE" == "strict" ]]; then
    ARTIFACT_PREFIX="training_beta025"
else
    ARTIFACT_PREFIX="analysis_beta025"
fi
CANONICAL_ARTIFACTS=(
    "$D1_DIR/$ARTIFACT_PREFIX.jsonl"
    "$D1_DIR/$ARTIFACT_PREFIX.all.jsonl"
    "$D1_DIR/$ARTIFACT_PREFIX.jsonl.manifest.json"
    "$D1_DIR/$ARTIFACT_PREFIX.failed-collection/failure_evidence.json"
)
for artifact_path in "${CANONICAL_ARTIFACTS[@]}"; do
    if [[ -e "$artifact_path" || -L "$artifact_path" ]]; then
        echo "ERROR: refusing to attribute a pre-existing D1 artifact to this attempt: $artifact_path" >&2
        exit 45
    fi
done
if [[ -L "$D1_DIR/$ARTIFACT_PREFIX.failed-collection" ]]; then
    echo "ERROR: refusing a symlinked D1 failure-evidence directory." >&2
    exit 45
fi
ARTIFACT_OWNERSHIP_ESTABLISHED=1

begin_signal_defer
COLLECTION_INTERVAL_START_EPOCH=$(date +%s.%N)
setsid nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
    --format=csv -l 1 > "$TELEMETRY" &
NVML_PID=$!
NVML_PGID=$NVML_PID
NVML_PROCESS_QUIESCED=0
end_signal_defer

export PY REPO COMMIT D0_DIR D1_DIR LEARNER REV HFROOT MODE BETA
export RESTART_COUNT
COLLECTION_WRAPPER="$REPO/scripts/run_dagger1_collection.sh"
export COLLECTION_WRAPPER COLLECTION_SUPERVISOR_STATUS
export COLLECTION_TREE_QUIESCED_MARKER
set +e
begin_signal_defer
# The Python session leader is a Linux child subreaper.  A collector descendant
# may call setsid(2) and escape the original process group, but it remains in
# this supervisor's descendant tree and is adopted here if its parent exits.
# The supervisor does not exit until that entire tree is TERM/KILL-quiesced.
setsid "$PY" - <<'PY' &
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time


PR_SET_CHILD_SUBREAPER = 36
TERM_GRACE_SECONDS = 0.5
KILL_GRACE_SECONDS = 1.0

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    error_number = ctypes.get_errno()
    raise SystemExit(
        f"PR_SET_CHILD_SUBREAPER failed: {os.strerror(error_number)}"
    )

observed_signal: int | None = None


def remember_signal(signum: int, _frame: object) -> None:
    global observed_signal
    if observed_signal is None:
        observed_signal = signum


signal.signal(signal.SIGINT, remember_signal)
signal.signal(signal.SIGTERM, remember_signal)


def child_pids(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(
            encoding="ascii"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [int(value) for value in text.split()]


def descendants() -> set[int]:
    found: set[int] = set()
    pending = child_pids(os.getpid())
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(child_pids(pid))
    return found


def process_is_live(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    closing = stat.rfind(")")
    return closing >= 0 and stat[closing + 2 : closing + 3] != "Z"


def reap_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def live_descendants() -> set[int]:
    reap_children()
    return {pid for pid in descendants() if process_is_live(pid)}


def drain_descendant_tree(signum: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        targets = live_descendants()
        if not targets:
            return True
        for pid in sorted(targets, reverse=True):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        time.sleep(0.05)
    return not live_descendants()


def publish_no_replace(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


wrapper_environment = os.environ.copy()
for internal_name in (
    "COLLECTION_SUPERVISOR_STATUS",
    "COLLECTION_TREE_QUIESCED_MARKER",
    "COLLECTION_WRAPPER",
):
    wrapper_environment.pop(internal_name, None)
wrapper = subprocess.Popen(
    ["bash", os.environ["COLLECTION_WRAPPER"]],
    env=wrapper_environment,
)
wrapper_returncode: int | None = None
while wrapper_returncode is None and observed_signal is None:
    wrapper_returncode = wrapper.poll()
    if wrapper_returncode is None:
        time.sleep(0.05)

wrapper_completed_before_signal = wrapper_returncode is not None

kill_escalated = not drain_descendant_tree(
    signal.SIGTERM, TERM_GRACE_SECONDS
)
if kill_escalated:
    tree_quiesced = drain_descendant_tree(
        signal.SIGKILL, KILL_GRACE_SECONDS
    )
else:
    tree_quiesced = True

if wrapper_returncode is None:
    wrapper_returncode = wrapper.poll()
if observed_signal is not None:
    collection_exit_code = 128 + observed_signal
elif wrapper_returncode is None:
    collection_exit_code = 125
elif wrapper_returncode < 0:
    collection_exit_code = 128 - wrapper_returncode
else:
    collection_exit_code = wrapper_returncode
if not 0 <= collection_exit_code <= 255:
    collection_exit_code = 125

status = {
    "contract": "dagger1_collection_subreaper_v2",
    "supervisor_pid": os.getpid(),
    "wrapper_pid": wrapper.pid,
    "wrapper_returncode": wrapper_returncode,
    "wrapper_completed_before_signal": wrapper_completed_before_signal,
    "collection_exit_code": collection_exit_code,
    "termination_signal": (
        signal.Signals(observed_signal).name if observed_signal is not None else None
    ),
    "kill_escalated": kill_escalated,
    "descendant_tree_quiesced": tree_quiesced,
}
serialized = (json.dumps(status, indent=2, sort_keys=True) + "\n").encode()
status_path = Path(os.environ["COLLECTION_SUPERVISOR_STATUS"])
try:
    if not tree_quiesced:
        raise SystemExit(125)
    publish_no_replace(status_path, serialized)
    publish_no_replace(
        Path(os.environ["COLLECTION_TREE_QUIESCED_MARKER"]),
        hashlib.sha256(serialized).hexdigest().encode() + b"\n",
    )
except (OSError, RuntimeError):
    raise SystemExit(124)

# The outer launcher never interprets this as the collection result.  This one
# reserved code authenticates only that this exact supervisor successfully
# published and fsynced its hash-bound, fully-quiesced transport evidence.
raise SystemExit(86)
PY
COLLECT_PID=$!
COLLECT_PGID=$COLLECT_PID
COLLECT_SUPERVISOR_PID=$COLLECT_PID
COLLECTION_STARTED=1
COLLECTION_PROCESS_STATUS="running"
COLLECTION_PROCESS_QUIESCED=0
end_signal_defer
wait "$COLLECT_PID"
SUPERVISOR_TRANSPORT_RC=$?
set -e
COLLECT_RC=$SUPERVISOR_TRANSPORT_FAILURE_EXIT_CODE
COLLECTION_COMPLETED=0
COLLECTION_PROCESS_EXIT_CODE=$SUPERVISOR_TRANSPORT_RC
COLLECT_PID=""
if process_group_has_live_members "$COLLECT_PGID"; then
    # The wrapper exited but a descendant survived.  Quiesce the isolated
    # session before any artifact hash can be computed.
    terminate_collection_group TERM
else
    COLLECTION_PROCESS_STATUS="completed"
    COLLECTION_PROCESS_QUIESCED=1
    apply_collection_supervisor_status
    if [[ "$COLLECTION_PROCESS_QUIESCED" -eq 1 ]]; then
        COLLECT_PGID=""
    fi
fi
mark_collection_interval_end
exit "$COLLECT_RC"

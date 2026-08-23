"""Regression tests for the fail-closed DAgger-1 Slurm launcher."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import unittest


class Dagger1CollectionLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_path = (
            Path(__file__).resolve().parents[2] / "submit_dagger1_collection.sh"
        )
        cls.launcher_bytes = cls.launcher_path.read_bytes()
        cls.launcher = cls.launcher_bytes.decode("utf-8")
        cls.collection_wrapper = (
            cls.launcher_path.parent / "scripts" / "run_dagger1_collection.sh"
        ).read_text(encoding="utf-8")
        cls.bash = shutil.which("bash")
        if cls.bash is not None:
            cls.bash_path = subprocess.run(
                [cls.bash, "-lc", 'printf "%s" "$PATH"'],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

    def setUp(self) -> None:
        if self.bash is None:
            self.skipTest("bash is required for executable launcher tests")
        self.owner = tempfile.TemporaryDirectory()
        self.root = Path(self.owner.name).resolve()
        self.repo = self.root / "repo"
        self.d0 = self.root / "d0"
        self.d1 = self.root / "d1"
        self.learner = self.root / "learner"
        self.hfroot = self.root / "hf"
        self.bin = self.root / "bin"
        for directory in (
            self.repo / "scripts",
            self.d0,
            self.d1,
            self.learner,
            self.hfroot,
            self.bin,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.launcher_copy = self.root / "submit_dagger1_collection.sh"
        self.launcher_copy.write_text(
            self.launcher,
            encoding="utf-8",
            newline="\n",
        )
        self._write_executable(
            self.bin / "python-stub",
            """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "psse_env.sft.release_hardware" ]]; then
    printf '%s\n' '{"device_count":1,"devices":[{"index":0,"name":"NVIDIA H200","total_memory_bytes":150754820096,"accelerator_class":"h200"}]}'
    exit "${HARDWARE_EXIT_CODE:-0}"
fi
if [[ "${1:-}" == "-" ]]; then
    exec python3 "$@"
fi
echo "unexpected python stub arguments: $*" >&2
exit 97
""",
        )
        self._write_executable(
            self.bin / "scontrol",
            """#!/bin/bash
printf 'JobId=%s NodeList=stub-node\n' "${SLURM_JOB_ID:?}"
""",
        )
        self._write_executable(
            self.bin / "ps",
            """#!/bin/bash
/usr/bin/ps "$@"
ps_rc=$?
if [[ "${MONITOR_MODE:-}" == "unverifiable_cleanup" \
    && -s "$D1_DIR/nvml_sticky.pgid" ]]; then
    printf '%s S\n' "$(< "$D1_DIR/nvml_sticky.pgid")"
fi
exit "$ps_rc"
""",
        )
        self._write_executable(
            self.bin / "mkdir",
            """#!/bin/bash
/usr/bin/mkdir "$@"
mkdir_rc=$?
last_argument=${!#}
if [[ "$mkdir_rc" -eq 0 && "${SIGNAL_DURING_CLAIM:-0}" == "1" \
    && "$last_argument" == */attempt-* ]]; then
    kill -TERM "$PPID"
fi
exit "$mkdir_rc"
""",
        )
        self._write_executable(
            self.bin / "rmdir",
            """#!/bin/bash
target=${!#}
if [[ "${FAIL_LOCK_RELEASE:-0}" == "1" \
    && "$target" == */.dagger1_collection_owners/strict ]]; then
    exit 19
fi
exec /usr/bin/rmdir "$@"
""",
        )
        self._write_executable(
            self.bin / "date",
            """#!/bin/bash
delay_marker="$D1_DIR/.collection_interval_date_delayed"
if [[ "${DELAY_COLLECTION_INTERVAL_DATE:-0}" == "1" \
    && "$*" == "+%s.%N" && ! -e "$delay_marker" ]]; then
    : > "$delay_marker"
    sleep 4
fi
exec /usr/bin/date "$@"
""",
        )
        self._write_executable(
            self.bin / "nvidia-smi",
            """#!/bin/bash
set -u
header='timestamp, index, uuid, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W], clocks.sm [MHz]'
if [[ "$*" == *"--format=csv,noheader,nounits"* ]]; then
    printf '%s\n' 'GPU-a1b2, NVIDIA H200, 143771'
    exit 0
fi
now_timestamp() { date '+%Y/%m/%d %H:%M:%S.000'; }
case "${MONITOR_MODE:-good}" in
    good)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    summary|forged_fast)
        printf '%s\n' "$header"
        python3 - <<'PY'
from datetime import datetime, timedelta
start = datetime.now()
for second in range(60):
    utilization, memory, power = (
        (0, 1024, 100.0),
        (50, 2048, 200.0),
        (100, 3072, 300.0),
        (100, 4096, 400.0),
    )[second % 4]
    timestamp = (start + timedelta(seconds=second)).strftime(
        "%Y/%m/%d %H:%M:%S.%f"
    )[:-3]
    print(
        f"{timestamp}, 0, GPU-a1b2, NVIDIA H200, {utilization} %, "
        f"{memory} MiB, 143771 MiB, {power} W, 1800 MHz"
    )
PY
        ;;
    low)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 59 %%, 1024 MiB, 143771 MiB, 150.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    malformed_coverage)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 99 %%, 1024 MiB, 143771 MiB, 150.0 W, 1800 MHz\n' "$(now_timestamp)"
        for _ in $(seq 1 59); do printf '%s\n' 'malformed,row'; done
        ;;
    device_mismatch)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-forged, NVIDIA RTX 6000 Ada Generation, 99 %%, 1024 MiB, 49140 MiB, 150.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    name_mismatch)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H100, 99 %%, 1024 MiB, 143771 MiB, 150.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    memory_mismatch)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 99 %%, 1024 MiB, 143700 MiB, 150.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    mixed_uuid)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        printf '%s, 0, GPU-c3d4, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        ;;
    unverifiable_cleanup)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        /usr/bin/ps -o pgid= -p $$ | tr -d ' ' > "$D1_DIR/nvml_sticky.pgid"
        ;;
    stubborn)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        trap '' INT TERM
        ;;
    signal_finalize)
        printf '%s\n' "$header"
        printf '%s, 0, GPU-a1b2, NVIDIA H200, 75 %%, 1024 MiB, 143771 MiB, 350.0 W, 1800 MHz\n' "$(now_timestamp)"
        trap 'kill -TERM "$PPID"; exit 0' TERM
        ;;
    empty)
        printf '%s\n' "$header"
        ;;
    failing)
        exit 9
        ;;
    *)
        exit 98
        ;;
esac
: > "$D1_DIR/nvml_stub_ready"
if [[ "${MONITOR_MODE:-good}" != "stubborn" \
    && "${MONITOR_MODE:-good}" != "signal_finalize" ]]; then
    trap 'exit 0' INT TERM
fi
while :; do sleep 1; done
""",
        )
        self._write_executable(
            self.repo / "scripts" / "run_dagger1_collection.sh",
            """#!/bin/bash
set -euo pipefail
case "${COLLECTOR_MODE:-success}" in
    success)
        # Synchronize the fast collector stub with monitor initialization. A
        # loaded Linux runner may otherwise finish this 0.2-second collector
        # before the monitor process receives its first scheduling slice.
        for _ in $(seq 1 200); do
            [[ -e "$D1_DIR/nvml_stub_ready" ]] && break
            sleep 0.01
        done
        sleep 0.2
        ;;
    slow)
        # The lock-race test needs the winning launcher to remain alive long
        # enough to hold the mode lock, but it should not also depend on the
        # fake monitor winning an unrelated scheduling race on a loaded Linux
        # runner.  Synchronize with the same ready marker as the fast stub.
        for _ in $(seq 1 200); do
            [[ -e "$D1_DIR/nvml_stub_ready" ]] && break
            sleep 0.01
        done
        sleep 1
        ;;
    term|int)
        sleep 0.1
        signal_name=${COLLECTOR_MODE^^}
        kill -"$signal_name" "$PPID"
        sleep 5
        ;;
    stubborn_term)
        trap '' INT TERM
        (
            trap '' INT TERM
            while :; do sleep 1; done
        ) &
        printf '%s\n' "$!" > "$D1_DIR/stubborn_child.pid"
        kill -TERM "$PPID"
        while :; do sleep 1; done
        ;;
    leader_exit_stubborn_child)
        launcher_pid=$PPID
        (
            trap '' INT TERM
            sleep 0.05
            kill -TERM "$launcher_pid"
            while :; do sleep 1; done
        ) &
        printf '%s\n' "$!" > "$D1_DIR/post_leader_child.pid"
        exit 0
        ;;
    setsid_escape)
        setsid bash -c '
            trap "" INT TERM
            printf "%s\n" "$$" > "$D1_DIR/setsid_escape.pid"
            while :; do sleep 1; done
        ' &
        exit 0
        ;;
    forge_supervisor_evidence)
        if [[ -n "${COLLECTION_SUPERVISOR_STATUS+x}" \
            || -n "${COLLECTION_TREE_QUIESCED_MARKER+x}" ]]; then
            exit 95
        fi
        setsid bash -c '
            trap "" INT TERM
            printf "%s\n" "$$" > "$D1_DIR/forged_escape.pid"
            while :; do sleep 1; done
        ' </dev/null >/dev/null 2>&1 &
        status="$D1_DIR/run_receipts/$SLURM_JOB_ID/attempt-$SLURM_RESTART_COUNT/collection_supervisor_status.json"
        marker="$D1_DIR/run_receipts/$SLURM_JOB_ID/attempt-$SLURM_RESTART_COUNT/collection_tree_quiesced"
        printf '%s\n' '{"contract":"forged","descendant_tree_quiesced":true}' > "$status"
        python3 - "$status" "$marker" <<'PY'
import hashlib
from pathlib import Path
import sys
content = Path(sys.argv[1]).read_bytes()
Path(sys.argv[2]).write_text(hashlib.sha256(content).hexdigest() + "\\n")
PY
        supervisor_pid=$(/usr/bin/ps -o ppid= -p $$ | tr -d ' ')
        printf '%s\n' "$supervisor_pid" > "$D1_DIR/forged_supervisor.pid"
        kill -KILL "$supervisor_pid"
        exit 0
        ;;
    collide)
        destination="$D1_DIR/run_receipts/$SLURM_JOB_ID/attempt-$SLURM_RESTART_COUNT/run_receipt.json"
        printf 'preexisting-receipt\n' > "$destination"
        sleep 0.2
        ;;
    *)
        exit 96
        ;;
esac
""",
        )
        self.posix_root = self._posix_cwd(self.root)

    def tearDown(self) -> None:
        if hasattr(self, "owner"):
            self.owner.cleanup()

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def _posix_cwd(self, path: Path) -> str:
        return subprocess.run(
            [self.bash, "-lc", "pwd"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _run(
        self,
        *,
        job_id: str = "7001",
        restart_count: str = "0",
        monitor_mode: str = "good",
        collector_mode: str = "success",
        hardware_exit_code: str = "0",
        signal_during_claim: str = "0",
        signal_before_receipt_dir_fsync: str = "0",
        signal_publisher_before_receipt_dir_fsync: str = "0",
        kill_publisher_before_receipt_dir_fsync: str = "0",
        kill_publisher_after_final_unlink: str = "0",
        signal_immediately_before_launcher_exit: str = "0",
        fail_lock_release: str = "0",
        fail_receipt_dir_fsync: str = "0",
        fail_receipt_rollback_fsync: str = "0",
        fail_receipt_temp_unlink: str = "0",
        delay_collection_interval_date: str = "0",
    ) -> subprocess.CompletedProcess[bytes]:
        posix = self.posix_root
        assignments = {
            "PATH": f"{posix}/bin:{self.bash_path}",
            "PY": f"{posix}/bin/python-stub",
            "REPO": f"{posix}/repo",
            "COMMIT": "a" * 40,
            "D0_DIR": f"{posix}/d0",
            "D1_DIR": f"{posix}/d1",
            "LEARNER": f"{posix}/learner",
            "REV": "b" * 64,
            "HFROOT": f"{posix}/hf",
            "MODE": "strict",
            "BETA": "0.25",
            "EXPECTED_ACCELERATOR_CLASS": "h200",
            "SLURM_JOB_ID": job_id,
            "SLURM_RESTART_COUNT": restart_count,
            "MONITOR_MODE": monitor_mode,
            "COLLECTOR_MODE": collector_mode,
            "HARDWARE_EXIT_CODE": hardware_exit_code,
            "SIGNAL_DURING_CLAIM": signal_during_claim,
            "SIGNAL_BEFORE_RECEIPT_DIR_FSYNC": (
                signal_before_receipt_dir_fsync
            ),
            "SIGNAL_PUBLISHER_BEFORE_RECEIPT_DIR_FSYNC": (
                signal_publisher_before_receipt_dir_fsync
            ),
            "KILL_PUBLISHER_BEFORE_RECEIPT_DIR_FSYNC": (
                kill_publisher_before_receipt_dir_fsync
            ),
            "KILL_PUBLISHER_AFTER_FINAL_UNLINK_BEFORE_ROLLBACK_FSYNC": (
                kill_publisher_after_final_unlink
            ),
            "SIGNAL_IMMEDIATELY_BEFORE_LAUNCHER_EXIT": (
                signal_immediately_before_launcher_exit
            ),
            "FAIL_LOCK_RELEASE": fail_lock_release,
            "FAIL_RECEIPT_DIR_FSYNC": fail_receipt_dir_fsync,
            "FAIL_RECEIPT_ROLLBACK_FSYNC": fail_receipt_rollback_fsync,
            "FAIL_RECEIPT_TEMP_UNLINK": fail_receipt_temp_unlink,
            "DELAY_COLLECTION_INTERVAL_DATE": delay_collection_interval_date,
        }
        prefix = "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in assignments.items()
        )
        return subprocess.run(
            [self.bash, "-s"],
            cwd=self.root,
            input=(prefix + self.launcher).encode("utf-8"),
            check=False,
            capture_output=True,
            timeout=15,
        )

    def _receipt(self, job_id: str, restart_count: str = "0") -> dict:
        path = (
            self.d1
            / "run_receipts"
            / job_id
            / f"attempt-{restart_count}"
            / "run_receipt.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_actual_launcher_is_lf_utf8_and_bash_syntax_valid(self) -> None:
        self.assertFalse(self.launcher_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", self.launcher_bytes)
        syntax = subprocess.run(
            [self.bash, "-n", self.launcher_path.name],
            cwd=self.launcher_path.parent,
            check=False,
            capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr.decode())

    def test_scheduler_can_choose_hopper_or_high_memory_rtx(self) -> None:
        for contract in (
            "#SBATCH --gres=gpu:1",
            '#SBATCH --constraint="h200|h100|rtx6000"',
            '#SBATCH --comment="preemption=yes;requeue=true"',
            "#SBATCH --requeue",
            "#SBATCH --open-mode=append",
            "sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=128G",
            "EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}",
            '--require-class "$EXPECTED_ACCELERATOR_CLASS"',
            'cd "$REPO"',
        ):
            self.assertIn(contract, self.launcher)
        self.assertNotIn("#SBATCH --partition", self.launcher)
        self.assertNotIn("#SBATCH --qos", self.launcher)
        self.assertNotIn("#SBATCH --no-requeue", self.launcher)

    def test_launch_is_provenance_bound_and_telemetry_backed(self) -> None:
        for contract in (
            'COLLECTION_WRAPPER="$REPO/scripts/run_dagger1_collection.sh"',
            "COMMIT must be the reviewed 40-hex source commit",
            "REV must be the exact 64-hex learner adapter tree digest",
            "the canonical DAgger-1 launcher requires BETA=0.25",
            "dagger1_slurm_run_receipt_v3",
            '"collection_wrapper_exit_code"',
            '"launcher_exit_code"',
            "accelerator_attestation.json",
            "nvml_device_attestation.csv",
            "nvml_device_attestation_sha256",
            "SLURM_JOB_PARTITION",
            "SLURM_JOB_QOS",
            "SLURM_JOB_GRES",
            "SLURM_JOB_CONSTRAINTS",
            "nvml_1s.csv",
            "--query-gpu=timestamp,index,uuid,name,utilization.gpu",
            '"nvml_monitor_exit_code"',
            '"nvml_telemetry_hashing_skipped"',
            '"nvml_valid_sample_count"',
            '"total_data_rows"',
            '"invalid_data_rows"',
            '"timestamp_span_seconds"',
            '"max_timestamp_gap_seconds"',
            '"representative_coverage"',
            '"collection_interval_coverage"',
            '"device_identity_match"',
            '"mean_gpu_utilization_percent"',
            '"nonzero_gpu_duty_percent"',
            '"p95_gpu_utilization_percent"',
            '"mean_power_draw_watts"',
            '"max_memory_used_mib"',
            '"nyu_gpu_utilization_classification"',
            "75% warning",
            "60% cancellation-risk",
            "classifications only",
            "PR_SET_CHILD_SUBREAPER",
            'setsid "$PY" -',
            'kill -KILL -- "-$pgid"',
            "os.fsync(handle.fileno())",
            "os.fsync(directory_fd)",
            "signal.signal(signal.SIGINT, raise_publisher_signal)",
            "signal.signal(signal.SIGTERM, raise_publisher_signal)",
        ):
            self.assertIn(contract, self.launcher)
        self.assertNotIn('"collector_exit_code"', self.launcher)

    def test_canonical_wrapper_enables_overlap_policy_audit(self) -> None:
        self.assertEqual(
            self.collection_wrapper.count("--overlap-policy-audit"),
            1,
        )
        self.assertIn(
            "canonical collection wrapper enables --overlap-policy-audit",
            self.launcher,
        )

    def test_job_receipt_directory_is_claimed_atomically(self) -> None:
        self.assertIn("SLURM_RESTART_COUNT", self.launcher)
        self.assertIn('attempt-$RESTART_COUNT', self.launcher)
        self.assertIn('if ! mkdir -- "$RECEIPT_DIR"', self.launcher)
        self.assertNotIn('if [[ -e "$RECEIPT_DIR" ]]', self.launcher)
        self.assertIn("refusing to reuse run receipt directory", self.launcher)
        self.assertLess(
            self.launcher.index("trap on_exit EXIT"),
            self.launcher.index('if ! mkdir -- "$RECEIPT_DIR"'),
        )
        self.assertLess(
            self.launcher.index("install_runtime_signal_traps\n    signal_name="),
            self.launcher.index("signal_name=$DEFERRED_SIGNAL_NAME"),
        )

    def test_signal_during_atomic_claim_publishes_owned_receipt(self) -> None:
        result = self._run(job_id="7016", signal_during_claim="1")
        receipt = self._receipt("7016")
        self.assertEqual(result.returncode, 143, result.stderr.decode())
        self.assertEqual(receipt["termination_signal"], "TERM")
        self.assertFalse(receipt["collection_started"])
        self.assertEqual(receipt["launcher_exit_code"], 143)

    def test_signal_during_finalization_is_deferred_through_receipt_fsync(self) -> None:
        result = self._run(job_id="7018", monitor_mode="signal_finalize")
        receipt = self._receipt("7018")
        self.assertEqual(result.returncode, 143, result.stderr.decode())
        self.assertEqual(receipt["termination_signal"], "TERM")
        self.assertEqual(receipt["launcher_exit_code"], 143)
        self.assertTrue(receipt["collection_process_quiesced"])
        self.assertIn(
            receipt["nvml_monitor_status"],
            {"terminated_by_launcher", "exited_before_finalize"},
        )

    def test_post_freeze_term_cannot_diverge_process_and_receipt_status(self) -> None:
        result = self._run(
            job_id="7025",
            signal_before_receipt_dir_fsync="1",
        )
        receipt = self._receipt("7025")
        self.assertEqual(receipt["launcher_exit_code"], result.returncode)
        self.assertNotEqual(result.returncode, 143, result.stderr.decode())
        self.assertIsNone(receipt["termination_signal"])
        self.assertEqual(
            receipt["terminal_state_freeze_contract"],
            "signals_observed_after_terminal_state_freeze_do_not_change_receipted_exit_status",
        )
        self.assertIn(b"after immutable terminal-state freeze", result.stderr)

    def test_term_immediately_before_exit_uses_receipted_status(self) -> None:
        result = self._run(
            job_id="7029",
            signal_immediately_before_launcher_exit="1",
        )
        receipt = self._receipt("7029")
        self.assertEqual(receipt["launcher_exit_code"], result.returncode)
        self.assertNotEqual(result.returncode, 143, result.stderr.decode())
        self.assertIsNone(receipt["termination_signal"])
        self.assertIn(b"after immutable terminal-state freeze", result.stderr)
        self.assertNotIn("trap - INT TERM", self.launcher)

    def test_post_receipt_lock_release_failure_preserves_receipted_status(self) -> None:
        result = self._run(job_id="7030", fail_lock_release="1")
        receipt = self._receipt("7030")
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        self.assertEqual(receipt["launcher_exit_code"], result.returncode)
        self.assertNotEqual(result.returncode, 46, result.stderr.decode())
        self.assertTrue(lock.is_dir())
        self.assertIn(b"failed to release the owned D1 mode lock", result.stderr)

    def test_post_link_fsync_failure_rolls_back_only_owned_receipt(self) -> None:
        result = self._run(job_id="7031", fail_receipt_dir_fsync="1")
        receipt_dir = self.d1 / "run_receipts" / "7031" / "attempt-0"
        self.assertEqual(result.returncode, 42, result.stderr.decode())
        self.assertFalse((receipt_dir / "run_receipt.json").exists())
        self.assertFalse(list(receipt_dir.glob(".run_receipt.*.tmp")))
        self.assertIn(b"owned final link rolled back", result.stderr)

    def test_indeterminate_receipt_rollback_retains_mode_lock(self) -> None:
        result = self._run(
            job_id="7033",
            fail_receipt_dir_fsync="1",
            fail_receipt_rollback_fsync="1",
        )
        receipt_dir = self.d1 / "run_receipts" / "7033" / "attempt-0"
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        self.assertEqual(result.returncode, 47, result.stderr.decode())
        self.assertFalse((receipt_dir / "run_receipt.json").exists())
        self.assertTrue(lock.is_dir())
        self.assertIn(b"publication is indeterminate", result.stderr)
        self.assertIn(b"rollback durability is indeterminate", result.stderr)

    def test_persistent_temp_unlink_failure_still_rolls_back_final_link(self) -> None:
        result = self._run(job_id="7034", fail_receipt_temp_unlink="1")
        receipt_dir = self.d1 / "run_receipts" / "7034" / "attempt-0"
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        temporary_files = list(receipt_dir.glob(".run_receipt.*.tmp"))
        try:
            self.assertEqual(result.returncode, 47, result.stderr.decode())
            self.assertFalse((receipt_dir / "run_receipt.json").exists())
            self.assertEqual(len(temporary_files), 1)
            self.assertTrue(lock.is_dir())
            self.assertIn(b"temporary cleanup is indeterminate", result.stderr)
            self.assertIn(b"publication is indeterminate", result.stderr)
        finally:
            for path in temporary_files:
                path.chmod(0o600)
                path.unlink(missing_ok=True)

    def test_publisher_term_before_directory_fsync_rolls_back_owned_link(self) -> None:
        result = self._run(
            job_id="7035",
            signal_publisher_before_receipt_dir_fsync="1",
        )
        receipt_dir = self.d1 / "run_receipts" / "7035" / "attempt-0"
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        self.assertEqual(result.returncode, 42, result.stderr.decode())
        self.assertFalse((receipt_dir / "run_receipt.json").exists())
        self.assertFalse(list(receipt_dir.glob(".run_receipt.*.tmp")))
        self.assertFalse(lock.exists())
        self.assertIn(b"owned final link rolled back", result.stderr)

    def test_abnormal_publisher_exit_with_visible_final_is_indeterminate(self) -> None:
        result = self._run(
            job_id="7036",
            kill_publisher_before_receipt_dir_fsync="1",
        )
        receipt_dir = self.d1 / "run_receipts" / "7036" / "attempt-0"
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        self.assertEqual(result.returncode, 47, result.stderr.decode())
        self.assertTrue((receipt_dir / "run_receipt.json").is_file())
        self.assertFalse(list(receipt_dir.glob(".run_receipt.*.tmp")))
        self.assertTrue(lock.is_dir())
        self.assertIn(b"publication is indeterminate", result.stderr)
        self.assertIn(b"rollback durability is indeterminate", result.stderr)

    def test_publisher_death_after_unlink_without_fsync_is_indeterminate(self) -> None:
        result = self._run(
            job_id="7037",
            fail_receipt_dir_fsync="1",
            kill_publisher_after_final_unlink="1",
        )
        receipt_dir = self.d1 / "run_receipts" / "7037" / "attempt-0"
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        self.assertEqual(result.returncode, 47, result.stderr.decode())
        self.assertFalse((receipt_dir / "run_receipt.json").exists())
        self.assertTrue(lock.is_dir())
        self.assertIn(b"no authenticated clean-rollback", result.stderr)
        self.assertIn(b"rollback durability is indeterminate", result.stderr)

    def test_exit_traps_are_installed_for_failure_and_preemption(self) -> None:
        for contract in (
            "trap on_exit EXIT",
            "trap 'on_signal INT 130' INT",
            "trap 'on_signal TERM 143' TERM",
        ):
            self.assertIn(contract, self.launcher)

        result = self._run(job_id="7009", hardware_exit_code="17")
        receipt = self._receipt("7009")
        self.assertEqual(result.returncode, 17, result.stderr.decode())
        self.assertFalse(receipt["collection_started"])
        self.assertFalse(receipt["collection_completed"])
        self.assertIsNone(receipt["collection_wrapper_exit_code"])
        self.assertEqual(receipt["launcher_exit_code"], 17)
        self.assertEqual(receipt["nvml_monitor_status"], "not_started")

    def test_receipt_publication_is_atomic_and_no_replace(self) -> None:
        self.assertIn("os.link(temporary, final)", self.launcher)
        self.assertNotIn("os.replace(", self.launcher)
        result = self._run(job_id="7002", collector_mode="collide")
        destination = (
            self.d1
            / "run_receipts"
            / "7002"
            / "attempt-0"
            / "run_receipt.json"
        )
        self.assertEqual(result.returncode, 42, result.stderr.decode())
        self.assertEqual(destination.read_bytes(), b"preexisting-receipt\n")
        self.assertFalse(
            (self.d1 / ".dagger1_collection_owners" / "strict").exists()
        )
        self.assertIn(b"pre-existing receipt was preserved", result.stderr)
        self.assertNotIn(b"publication is indeterminate", result.stderr)

    def test_int_and_term_always_publish_termination_receipts(self) -> None:
        for job_id, mode, signal_name, signal_exit_code in (
            ("7003", "term", "TERM", 143),
            ("7010", "int", "INT", 130),
        ):
            with self.subTest(signal_name=signal_name):
                result = self._run(job_id=job_id, collector_mode=mode)
                receipt = self._receipt(job_id)
                self.assertEqual(
                    result.returncode,
                    signal_exit_code,
                    result.stderr.decode(),
                )
                self.assertEqual(receipt["termination_signal"], signal_name)
                self.assertEqual(receipt["launcher_exit_code"], signal_exit_code)
                self.assertEqual(
                    receipt["collection_wrapper_exit_code"],
                    signal_exit_code,
                )
                self.assertTrue(receipt["collection_started"])
                self.assertFalse(receipt["collection_completed"])

    def test_empty_and_failing_monitor_fail_closed(self) -> None:
        for job_id, monitor_mode, expected_status, expected_monitor_rc in (
            ("7004", "empty", "terminated_by_launcher", 0),
            ("7005", "failing", "exited_before_finalize", 9),
        ):
            with self.subTest(monitor_mode=monitor_mode):
                result = self._run(job_id=job_id, monitor_mode=monitor_mode)
                receipt = self._receipt(job_id)
                self.assertEqual(result.returncode, 40, result.stderr.decode())
                self.assertEqual(receipt["collection_wrapper_exit_code"], 0)
                self.assertEqual(receipt["launcher_exit_code"], 40)
                self.assertEqual(receipt["nvml_valid_sample_count"], 0)
                self.assertFalse(receipt["nvml_telemetry_requirement_met"])
                self.assertEqual(
                    receipt["nyu_gpu_utilization_classification"],
                    "unavailable",
                )
                self.assertEqual(receipt["nvml_monitor_status"], expected_status)
                self.assertEqual(
                    receipt["nvml_monitor_exit_code"], expected_monitor_rc
                )

    def test_telemetry_device_identity_is_bound_to_release_attestation(self) -> None:
        for job_id, monitor_mode in (
            ("7019", "device_mismatch"),
            ("7020", "mixed_uuid"),
            ("7026", "name_mismatch"),
            ("7027", "memory_mismatch"),
        ):
            with self.subTest(monitor_mode=monitor_mode):
                result = self._run(job_id=job_id, monitor_mode=monitor_mode)
                receipt = self._receipt(job_id)
                summary = receipt["nvml_telemetry_summary"]
                self.assertEqual(result.returncode, 40, result.stderr.decode())
                self.assertFalse(summary["device_identity_match"])
                self.assertGreaterEqual(summary["identity_mismatch_rows"], 1)
                self.assertFalse(receipt["nvml_telemetry_requirement_met"])

    def test_valid_monitor_allows_success(self) -> None:
        result = self._run(job_id="7006")
        receipt = self._receipt("7006")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertGreaterEqual(receipt["nvml_valid_sample_count"], 1)
        self.assertTrue(receipt["nvml_telemetry_requirement_met"])
        self.assertEqual(receipt["nvml_monitor_status"], "terminated_by_launcher")
        self.assertEqual(
            receipt["nvml_telemetry_summary"][
                "mean_gpu_utilization_percent"
            ],
            75.0,
        )
        self.assertEqual(
            receipt["nyu_gpu_utilization_classification"],
            "insufficient_representative_coverage",
        )

    def test_delayed_interval_clock_is_recorded_before_monitor_start(self) -> None:
        result = self._run(
            job_id="7032",
            delay_collection_interval_date="1",
        )
        receipt = self._receipt("7032")
        summary = receipt["nvml_telemetry_summary"]
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertTrue(summary["collection_interval_coverage"])
        self.assertGreaterEqual(summary["first_sample_start_offset_seconds"], -3.0)
        self.assertTrue(receipt["nvml_telemetry_requirement_met"])

    def test_forged_fast_trace_cannot_claim_representative_coverage(self) -> None:
        warning_result = self._run(job_id="7011", monitor_mode="summary")
        warning_receipt = self._receipt("7011")
        summary = warning_receipt["nvml_telemetry_summary"]
        self.assertEqual(
            warning_result.returncode,
            40,
            warning_result.stderr.decode(),
        )
        self.assertEqual(summary["mean_gpu_utilization_percent"], 62.5)
        self.assertEqual(summary["nonzero_gpu_duty_percent"], 75.0)
        self.assertEqual(summary["p95_gpu_utilization_percent"], 100.0)
        self.assertEqual(summary["mean_power_draw_watts"], 250.0)
        self.assertEqual(summary["max_memory_used_mib"], 4096.0)
        self.assertEqual(summary["total_data_rows"], 60)
        self.assertEqual(summary["invalid_data_rows"], 0)
        self.assertEqual(summary["valid_data_rows"], 60)
        self.assertEqual(summary["timestamp_span_seconds"], 59.0)
        self.assertEqual(summary["max_timestamp_gap_seconds"], 1.0)
        self.assertFalse(summary["collection_interval_coverage"])
        self.assertFalse(summary["representative_coverage"])
        self.assertEqual(
            warning_receipt["nyu_gpu_utilization_classification"],
            "insufficient_representative_coverage",
        )
        self.assertEqual(
            warning_receipt["nyu_gpu_utilization_thresholds_percent"],
            {"warning": 75.0, "cancellation_risk": 60.0},
        )

        low_result = self._run(job_id="7012", monitor_mode="low")
        low_receipt = self._receipt("7012")
        self.assertEqual(low_result.returncode, 0, low_result.stderr.decode())
        self.assertTrue(low_receipt["nvml_telemetry_requirement_met"])
        self.assertEqual(
            low_receipt["nyu_gpu_utilization_classification"],
            "insufficient_representative_coverage",
        )

    def test_malformed_rows_cannot_claim_representative_utilization(self) -> None:
        result = self._run(job_id="7013", monitor_mode="malformed_coverage")
        receipt = self._receipt("7013")
        summary = receipt["nvml_telemetry_summary"]
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(summary["total_data_rows"], 60)
        self.assertEqual(summary["valid_data_rows"], 1)
        self.assertEqual(summary["invalid_data_rows"], 59)
        self.assertFalse(summary["representative_coverage"])
        self.assertEqual(
            receipt["nyu_gpu_utilization_classification"],
            "insufficient_representative_coverage",
        )

    def test_stubborn_monitor_is_killed_with_bounded_cleanup(self) -> None:
        result = self._run(job_id="7014", monitor_mode="stubborn")
        receipt = self._receipt("7014")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(receipt["nvml_monitor_status"], "killed_after_timeout")
        self.assertEqual(receipt["nvml_monitor_exit_code"], 137)
        self.assertTrue(receipt["nvml_process_quiesced"])
        self.assertTrue(receipt["nvml_kill_escalated"])
        self.assertFalse(receipt["nvml_telemetry_hashing_skipped"])
        self.assertIsNotNone(receipt["nvml_telemetry_sha256"])

    def test_unverified_monitor_quiescence_cannot_bless_racy_evidence(self) -> None:
        result = self._run(job_id="7017", monitor_mode="unverifiable_cleanup")
        receipt = self._receipt("7017")
        self.assertEqual(result.returncode, 40, result.stderr.decode())
        self.assertEqual(receipt["nvml_monitor_status"], "kill_timeout")
        self.assertFalse(receipt["nvml_process_quiesced"])
        self.assertTrue(receipt["nvml_kill_escalated"])
        self.assertTrue(receipt["nvml_telemetry_hashing_skipped"])
        self.assertIsNone(receipt["nvml_telemetry_sha256"])
        self.assertEqual(receipt["nvml_valid_sample_count"], 0)
        self.assertFalse(receipt["nvml_telemetry_requirement_met"])

    def test_stubborn_collector_tree_is_killed_before_receipt(self) -> None:
        result = self._run(job_id="7015", collector_mode="stubborn_term")
        receipt = self._receipt("7015")
        child_pid = int((self.d1 / "stubborn_child.pid").read_text().strip())
        child_probe = subprocess.run(
            [self.bash, "-lc", f"kill -0 {child_pid} 2>/dev/null"],
            check=False,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 143, result.stderr.decode())
        self.assertEqual(receipt["collection_process_status"], "killed_after_timeout")
        self.assertEqual(receipt["collection_process_exit_code"], 137)
        self.assertTrue(receipt["collection_process_quiesced"])
        self.assertTrue(receipt["collection_kill_escalated"])
        self.assertFalse(receipt["collection_artifact_hashing_skipped"])
        self.assertNotEqual(child_probe.returncode, 0)

    def test_post_leader_exit_signal_still_quiesces_stubborn_descendant(self) -> None:
        result = self._run(
            job_id="7021",
            collector_mode="leader_exit_stubborn_child",
        )
        receipt = self._receipt("7021")
        child_pid = int((self.d1 / "post_leader_child.pid").read_text().strip())
        child_probe = subprocess.run(
            [self.bash, "-lc", f"kill -0 {child_pid} 2>/dev/null"],
            check=False,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 143, result.stderr.decode())
        self.assertEqual(receipt["termination_signal"], "TERM")
        self.assertTrue(receipt["collection_completed"])
        self.assertTrue(receipt["collection_process_quiesced"])
        self.assertTrue(receipt["collection_kill_escalated"])
        self.assertNotEqual(child_probe.returncode, 0)

    def test_setsid_escape_is_adopted_and_killed_before_receipt(self) -> None:
        result = self._run(job_id="7028", collector_mode="setsid_escape")
        receipt = self._receipt("7028")
        escaped_pid = int((self.d1 / "setsid_escape.pid").read_text().strip())
        escaped_probe = subprocess.run(
            [self.bash, "-lc", f"kill -0 {escaped_pid} 2>/dev/null"],
            check=False,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertTrue(receipt["collection_process_quiesced"])
        self.assertTrue(receipt["collection_kill_escalated"])
        self.assertIsNotNone(receipt["collection_supervisor_status_sha256"])
        self.assertNotEqual(escaped_probe.returncode, 0)

    def test_forged_supervisor_files_cannot_authenticate_killed_transport(self) -> None:
        result = self._run(
            job_id="7032",
            collector_mode="forge_supervisor_evidence",
        )
        receipt = self._receipt("7032")
        escaped_pid = int((self.d1 / "forged_escape.pid").read_text().strip())
        lock = self.d1 / ".dagger1_collection_owners" / "strict"
        try:
            escaped_probe = subprocess.run(
                [self.bash, "-lc", f"kill -0 {escaped_pid} 2>/dev/null"],
                check=False,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 48, result.stderr.decode())
            self.assertFalse(receipt["collection_process_quiesced"])
            self.assertEqual(
                receipt["collection_process_status"],
                "supervisor_transport_failure",
            )
            self.assertEqual(
                receipt["collection_supervisor_transport_exit_code"],
                137,
            )
            self.assertFalse(
                receipt["collection_supervisor_transport_authenticated"]
            )
            self.assertIsNone(receipt["collection_supervisor_status_sha256"])
            self.assertTrue(receipt["collection_artifact_hashing_skipped"])
            self.assertTrue(
                all(
                    digest is None
                    for digest in receipt["collection_artifact_sha256"].values()
                )
            )
            self.assertTrue(lock.is_dir())
            self.assertEqual(escaped_probe.returncode, 0)
            self.assertIn(
                b"retaining the D1 mode lock",
                result.stderr,
            )
        finally:
            subprocess.run(
                [self.bash, "-lc", f"kill -KILL -- -{escaped_pid} 2>/dev/null"],
                check=False,
                capture_output=True,
            )

    def test_distinct_job_ids_cannot_share_mode_scoped_d1_outputs(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self._run, job_id=job_id, collector_mode="slow")
                for job_id in ("7022", "7023")
            ]
            results = [future.result(timeout=15) for future in futures]
        self.assertEqual(sorted(result.returncode for result in results), [0, 44])
        receipts = [self._receipt(job_id) for job_id in ("7022", "7023")]
        self.assertEqual(
            sorted(receipt["collection_started"] for receipt in receipts),
            [False, True],
        )
        self.assertFalse((self.d1 / ".dagger1_collection_owners" / "strict").exists())

    def test_early_hardware_failure_never_hashes_stale_canonical_artifacts(self) -> None:
        stale = self.d1 / "training_beta025.jsonl"
        stale.write_text("stale\n", encoding="utf-8")
        result = self._run(job_id="7024", hardware_exit_code="17")
        receipt = self._receipt("7024")
        self.assertEqual(result.returncode, 17, result.stderr.decode())
        self.assertTrue(receipt["collection_artifact_hashing_skipped"])
        self.assertTrue(
            all(
                digest is None
                for digest in receipt["collection_artifact_sha256"].values()
            )
        )
        self.assertEqual(stale.read_text(encoding="utf-8"), "stale\n")

    def test_requeue_attempts_are_distinct_and_cannot_be_reused(self) -> None:
        first = self._run(job_id="7007", restart_count="0")
        second = self._run(job_id="7007", restart_count="1")
        first_path = (
            self.d1 / "run_receipts" / "7007" / "attempt-0" / "run_receipt.json"
        )
        original = first_path.read_bytes()
        repeated = self._run(job_id="7007", restart_count="0")
        self.assertEqual(first.returncode, 0, first.stderr.decode())
        self.assertEqual(second.returncode, 0, second.stderr.decode())
        self.assertEqual(self._receipt("7007", "0")["slurm_restart_count"], 0)
        self.assertEqual(self._receipt("7007", "1")["slurm_restart_count"], 1)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(first_path.read_bytes(), original)

    def test_unsafe_or_noncanonical_job_ids_fail_before_path_creation(self) -> None:
        for job_id in (
            "",
            "0",
            "01",
            "../7008",
            "7008/1",
            "7008_1",
            "184467440737095516160",
        ):
            with self.subTest(job_id=job_id):
                result = self._run(job_id=job_id)
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    b"canonical positive decimal job ID",
                    result.stderr,
                )
        self.assertFalse((self.d1 / "run_receipts").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

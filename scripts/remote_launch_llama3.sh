#!/usr/bin/env bash
# Run in the prepared worker environment. Only --launch starts training.
set -euo pipefail
BASE=${MUSAPROBE_BASE:-/data/musaprobe}
PACKAGE=${MUSAPROBE_PACKAGE_ROOT:-}
MASTER_ADDR=${MUSAPROBE_MASTER_ADDR:-}
MASTER_PORT=${MUSAPROBE_MASTER_PORT:-}
EXPECTED_HOSTNAME=${MUSAPROBE_EXPECTED_HOSTNAME:-}
PROFILE=statistics
EXIT_INTERVAL=2
STEPS=0
RUN_ID=
LAUNCH=0
INJECTION=
usage() {
    echo "Usage: $0 --run-id ID --master-addr ADDRESS --master-port PORT [--base DIR] [--package DIR] [--expected-hostname NAME] [--profile baseline|statistics|tensor] [--exit-interval 2|8] [--steps 0|2,3] [--injection FILE] [--launch]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE=${2:?}; shift 2;;
        --package) PACKAGE=${2:?}; shift 2;;
        --master-addr) MASTER_ADDR=${2:?}; shift 2;;
        --master-port) MASTER_PORT=${2:?}; shift 2;;
        --expected-hostname) EXPECTED_HOSTNAME=${2:?}; shift 2;;
        --profile) PROFILE=${2:?}; shift 2;;
        --exit-interval) EXIT_INTERVAL=${2:?}; shift 2;;
        --steps) STEPS=${2:?}; shift 2;;
        --run-id) RUN_ID=${2:?}; shift 2;;
        --injection) INJECTION=${2:?}; shift 2;;
        --launch) LAUNCH=1; shift;;
        --help|-h) usage; exit 0;;
        *) usage >&2; exit 2;;
    esac
done
[[ -n $MASTER_ADDR && $MASTER_PORT =~ ^[0-9]+$ ]] || { echo "Explicit master address and port are required" >&2; exit 2; }
[[ $BASE == /* && $BASE != / ]] || { echo "Base must be an absolute directory other than /" >&2; exit 2; }
BASE=$(realpath -m -- "$BASE")
PACKAGE=${PACKAGE:-$BASE/package}
[[ $PACKAGE == /* ]] || { echo "Package must be an absolute directory" >&2; exit 2; }
PACKAGE=$(realpath -m -- "$PACKAGE")
[[ $PROFILE == baseline || $PROFILE == statistics || $PROFILE == tensor ]] || { echo "Invalid profile" >&2; exit 2; }
[[ $EXIT_INTERVAL == 2 || $EXIT_INTERVAL == 8 ]] || { echo "Exit interval must be 2 or 8" >&2; exit 2; }
[[ $STEPS =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Steps must be comma-separated nonnegative integers" >&2; exit 2; }
[[ $RUN_ID =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$ ]] || { echo "Explicit safe run ID required" >&2; exit 2; }
[[ -z $INJECTION || $PROFILE != baseline ]] || { echo "Injection requires capture profile" >&2; exit 2; }
[[ -d $BASE/workspace && -f $PACKAGE/src/musaprobe/integrations/megatron_entry.py ]] || { echo "Prepare isolated workspace and upload package first" >&2; exit 1; }
python3 - "$BASE" "$MASTER_ADDR" "$MASTER_PORT" "$EXPECTED_HOSTNAME" "$STEPS" "$EXIT_INTERVAL" <<'PY'
import hashlib, json, pathlib, socket, sys
base_arg, address, port_arg, expected_hostname, steps_arg, exit_arg = sys.argv[1:]
base = pathlib.Path(base_arg)
port = int(port_arg)
assert 1 <= port <= 65535, 'Master port must be in 1..65535'
steps = [int(step) for step in steps_arg.split(',')]
assert len(steps) == len(set(steps)) and all(step < int(exit_arg) for step in steps), 'Capture steps must be unique and below exit interval'
if expected_hostname:
    assert socket.gethostname() == expected_hostname, 'Current hostname does not match expected hostname'
audit = json.loads((base / 'audit/source_manifest.json').read_text())
assert audit.get('adapter_schema') == 2, 'Workspace requires the parameterized adapter; prepare a new isolated base'
assert audit['base'] == str(base) and audit['workspace'] == str(base / 'workspace'), 'Prepared workspace/base mismatch'
assert audit['master_addr'] == address and audit['master_port'] == port, 'Master endpoint differs from the prepared manifest'
script = base / 'workspace/scripts/run_pretrain_llama3_full.sh'
assert hashlib.sha256(script.read_bytes()).hexdigest() == audit['patched_sha256'], 'Prepared launch script has changed; review before running'
PY
RUN_DIR=$BASE/runs/$RUN_ID
export MUSAPROBE_MASTER_ADDR=$MASTER_ADDR
export MUSAPROBE_MASTER_PORT=$MASTER_PORT
export MUSAPROBE_EXIT_INTERVAL=$EXIT_INTERVAL
export MUSAPROBE_PACKAGE_ROOT=$PACKAGE
export MUSAPROBE_RUN_ID=$RUN_ID
export MUSAPROBE_SOURCE_MANIFEST=$BASE/audit/source_manifest.json
export MUSAPROBE_ENABLED=1
export MUSAPROBE_VALIDATE_PP_RANKS=1
[[ $PROFILE != baseline ]] || export MUSAPROBE_ENABLED=0
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$BASE/tmp
export TMP=$BASE/tmp
export TEMP=$BASE/tmp
export XDG_CACHE_HOME=$BASE/cache/xdg
export TORCH_HOME=$BASE/cache/torch
export TORCH_EXTENSIONS_DIR=$BASE/cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$BASE/cache/torchinductor
export TORCH_COMPILE_DEBUG_DIR=$BASE/cache/torch_compile_debug
export TRITON_CACHE_DIR=$BASE/cache/triton
export HF_HOME=$BASE/cache/huggingface
export HF_DATASETS_CACHE=$BASE/cache/huggingface/datasets
export CUDA_CACHE_PATH=$BASE/cache/cuda
export MUSA_CACHE_PATH=$BASE/cache/musa
export WANDB_DIR=$RUN_DIR/wandb
export WANDB_CACHE_DIR=$BASE/cache/wandb
export WANDB_CONFIG_DIR=$BASE/cache/wandb_config
# Some dependencies consult cwd for relative artifacts. The patched script later
# enters the isolated workspace; neither HOME nor source-tree permissions change.
export MPLCONFIGDIR=$BASE/cache/matplotlib
if [[ $LAUNCH == 0 ]]; then
    FULL_DRY_RUN=1 bash "$BASE/workspace/scripts/run_pretrain_llama3_full.sh" "$RUN_ID"
    printf 'Profile: %s; steps: %s; injection: %s; no training started. Add --launch to execute.\n' "$PROFILE" "$STEPS" "${INJECTION:-disabled}"
    exit 0
fi
[[ ! -e $RUN_DIR ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$BASE/runs" "$BASE/tmp" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_COMPILE_DEBUG_DIR" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$CUDA_CACHE_PATH" "$MUSA_CACHE_PATH" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR"
# One lease for all musaProbe validation runs; never stop an unrelated job.
exec 9>"$BASE/runs/training.lock"
flock -n 9 || { echo "Another musaProbe training process holds the lease" >&2; exit 1; }
mkdir "$RUN_DIR"
exec > >(tee "$RUN_DIR/train.log") 2>&1
TRAIN_PID=
SIGNALLED=0
finish() {
    local rc=$?
    trap - EXIT
    printf '%s\n' "$rc" >"$RUN_DIR/exit_code"
    if [[ $SIGNALLED == 1 ]]; then printf 'interrupted\n'; elif [[ $rc == 0 ]]; then printf 'completed\n'; else printf 'failed\n'; fi >"$RUN_DIR/status"
    exit "$rc"
}
trap finish EXIT
on_signal() {
    trap '' HUP TERM INT
    SIGNALLED=1
    if [[ -n $TRAIN_PID ]]; then
        kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" || true
    fi
    exit 143
}
trap on_signal HUP TERM INT
printf 'preflight\n' >"$RUN_DIR/status"
python3 - "$BASE" "$PACKAGE" "$PROFILE" "$RUN_ID" "$INJECTION" "$STEPS" "$EXIT_INTERVAL" "$MASTER_ADDR" "$MASTER_PORT" <<'PY'
import hashlib, json, pathlib, shutil, socket, subprocess, sys
base, package, profile, run_id, injection, steps_arg, exit_arg, master_addr, master_port = sys.argv[1:]
base, package = pathlib.Path(base), pathlib.Path(package)
steps = [int(step) for step in steps_arg.split(',')]
assert len(steps) == len(set(steps)) and all(step < int(exit_arg) for step in steps), 'Capture steps must be unique and below exit interval'
assert shutil.disk_usage(base).free >= 500 * 1024**3, 'Need at least 500 GiB free before starting a training experiment'
# Read-only device check; refuse to compete with existing training.
gpus = json.loads(subprocess.check_output(['mthreads-gmi', '-q', '--json'], text=True))['GPU']
assert len(gpus) == 8, f'Expected 8 GPUs, found {len(gpus)}'
busy = [g['Index'] for g in gpus if g['FB Memory Usage']['Used'] != '0MiB']
assert not busy, f'GPUs already in use: {busy}'
code_hashes = {
    str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((package / 'src').rglob('*.py'))
}
(base / 'runs' / run_id / 'probe_source.json').write_text(json.dumps({
    'files_sha256': code_hashes, 'profile': profile, 'steps': steps,
    'exit_interval': int(exit_arg),
    'master_addr': master_addr, 'master_port': int(master_port),
    'hostname': socket.gethostname(),
}, indent=2)+'\n')
with socket.socket() as listener:
    listener.bind((master_addr, int(master_port)))
if profile != 'baseline':
    cfg = json.loads((package / 'configs' / f'llama3_{profile}.json').read_text())
    cfg.update(output_dir=str(base / 'captures'), run_id=run_id, steps=steps)
    (base / 'runs' / run_id / 'probe_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
if injection:
    path = pathlib.Path(injection).resolve(strict=True)
    assert path.is_relative_to(base), 'Injection file must reside under the selected base'
    (base / 'runs' / run_id / 'injection.json').write_bytes(path.read_bytes())
print('PREFLIGHT PASS: isolated workspace, explicit local endpoint, and 8 idle GPUs; review the prepared launch arguments for model settings.')
PY
if [[ $PROFILE != baseline ]]; then
    export MUSAPROBE_CONFIG=$RUN_DIR/probe_config.json
fi
if [[ -n $INJECTION ]]; then
    export MUSAPROBE_INJECTION=$RUN_DIR/injection.json
else
    unset MUSAPROBE_INJECTION || true
fi
printf 'running\n' >"$RUN_DIR/status"
printf '%s\n' "$$" >"$RUN_DIR/worker.pid"
printf 'Starting run=%s profile=%s exit_interval=%s injection=%s\n' "$RUN_ID" "$PROFILE" "$EXIT_INTERVAL" "${MUSAPROBE_INJECTION:-disabled}"
setsid bash "$BASE/workspace/scripts/run_pretrain_llama3_full.sh" "$RUN_ID" &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" >"$RUN_DIR/torchrun.pid"
if wait "$TRAIN_PID"; then
    exit 0
else
    exit "$?"
fi

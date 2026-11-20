#!/usr/bin/env bash
# Sequential jobs share the same two FSDP trainer GPUs and dedicated TP=2 rollout.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# != 5 ]]; then
    echo "Usage: $0 RUN_ID SFT_ADAPTER STAGE1_STEPS STAGE2_STEPS STAGE3_STEPS" >&2
    exit 2
fi

run_id=$1
adapter=$2
stage_steps=("$3" "$4" "$5")
[[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && -f $adapter/adapter_model.safetensors ]] || exit 2
for steps in "${stage_steps[@]}"; do
    [[ $steps =~ ^[1-9][0-9]*$ ]] || exit 2
done
(( stage_steps[0] >= 2 && stage_steps[1] >= 2 && stage_steps[2] >= 3 )) || exit 2

root="$PWD"
train="$root/data/research/inverser-vlm-v1/train"
base="/data/v-guoqingluo/models/Qwen3.5-9B"
queue="$root/data/research/$run_id"
rollout="${CAPO_ROLLOUT_URL:-http://127.0.0.1:18021}"
[[ $rollout =~ ^http://127\.0\.0\.1:[0-9]+$ ]] || exit 2
smoke="$root/data/research/${CAPO_FSDP_SMOKE:-capo-grpo-joint-fsdp-smoke-v2}"
training_py="$root/.venv-train/bin/python"
test -f "$smoke/summary.json"
"$training_py" - "$smoke" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocol = json.loads((root / "manifest.json").read_text())
receipt = json.loads((root / "summary.json").read_text())
if not (
    protocol["trainer_ranks"] == 2 and protocol["trainer_sharding"] == "FSDP2"
    and receipt["status"] == "complete" and receipt["steps"] == 7
    and receipt["reward_receipts"]["stage3_T5_optimizer_batches"] >= 1
):
    sys.exit("Two-rank TP=2 joint smoke must complete before queuing the study")
PY
total=$((${stage_steps[0]} + ${stage_steps[1]} + ${stage_steps[2]}))

export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
export HF_HOME="$root/.cache/huggingface" XDG_CACHE_HOME="$root/.cache" PYTHONPATH="$root/src"

mkdir "$queue"
log_dir="$root/artifacts/capo/$run_id"
mkdir -p "$log_dir"
ln -s "$log_dir" "$queue/logs"
sha256sum "$train/manifest.json" "$adapter/adapter_model.safetensors" \
    "$root/src/inverser/capo/capo_grpo.py" "$root/src/inverser/capo/capo_checkpoint.py" \
    "$root/scripts/capo_train_queue.sh" > "$queue/input-sha256.txt"
seeds=(${CAPO_SEEDS:-20261119 20261120 20261121})
printf 'seed,arm,steps,stage1,stage2,stage3\n' > "$queue/schedule.csv"
for seed in "${seeds[@]}"; do
    for arm in outcome uncoupled capo; do
        printf '%s,%s,%s,%s,%s,%s\n' "$seed" "$arm" "$total" "${stage_steps[@]}" \
            >> "$queue/schedule.csv"
    done
done
for seed in "${seeds[@]}"; do
    for arm in outcome uncoupled capo; do
        job="$queue/$arm-$seed"
        echo "Running $arm seed=$seed steps=$total"
        if .venv-train/bin/torchrun --standalone --nproc-per-node=2 --module inverser \
            train-capo-grpo --split "$train" --model-path "$base" --adapter "$adapter" \
            --output "$job" --mode "$arm" --task joint --seed "$seed" --steps "$total" \
            --stage-steps "${stage_steps[@]}" --fsdp --vllm-server-url "$rollout" \
            > "$log_dir/$arm-$seed.log" 2>&1 && test -f "$job/summary.json" \
            && test -f "$job/checkpoint/full_adapter/manifest.json"; then
            echo "$arm,$seed,complete" >> "$queue/completed.csv"
        else
            # A failed collective may poison the dedicated rollout engine; do not silently retry.
            echo "$arm,$seed,failed" >> "$queue/completed.csv"
            exit 1
        fi
    done
done

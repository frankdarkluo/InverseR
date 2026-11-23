#!/usr/bin/env bash
# Deploy locally on any assigned GPU (0..3) so 4 checkpoints can evaluate in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 RUN_ID MERGED_MODEL SERVED_MODEL_NAME [GPU_ID]" >&2
    exit 2
fi

run_id=$1
model=$2
name=$3
gpu_id="${4:-${CAPO_EVAL_GPU:-0}}"
[[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $name != *" "* && $gpu_id =~ ^[0-3]$ ]] || exit 2
root="$PWD"
[[ -f $model/manifest.json ]] || exit 2
model=$(realpath "$model")
port=$((18010 + gpu_id))
url="http://127.0.0.1:$port"
queue="$root/artifacts/capo/$run_id"
if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
    echo "Dedicated valid port $port on GPU $gpu_id already occupied" >&2
    exit 2
fi
mkdir -p "$queue"

export CUDA_VISIBLE_DEVICES="$gpu_id" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$root/cache/vllm" HF_HOME="$root/cache/huggingface"
export XDG_CACHE_HOME="$root/cache" TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor"
envs/main/bin/vllm serve "$model" --served-model-name "$name" --host 127.0.0.1 \
    --port "$port" --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.75 \
    --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching --reasoning-parser qwen3 \
    --enforce-eager > "$queue/server.log" 2>&1 &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null || true' EXIT
for ((waited=0; waited<180; waited+=5)); do
    if curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1; then
        break
    fi
    kill -0 "$server_pid"
    sleep 5
done
curl -fsS --max-time 5 "$url/health" >/dev/null
kill -0 "$server_pid"
revision="checkpoint-manifest-sha256:$(sha256sum "$model/manifest.json" | cut -d' ' -f1)"
# Checkpoints are selected on the valid panel prefix (manifest panel_prefix: 500 groups).
split="$root/data/research/inverser/valid"
count=1000
common=(
    --split "$split" --count "$count" --model "$name" --revision "$revision"
    --local-vllm-url "$url/v1" --thinking --thinking-token-budget 1024 --max-output-tokens 2048
    --workers 16
)
# 5,000 requests at most (forward + T4 + 3 T5 attempts per row) x ~3.9k tokens ~= 20M; 2x cap.
envs/main/bin/python -m inverser run-vlm-eval "${common[@]}" --tasks forward T4 T5 \
    --output "data/research/$run_id-valid" --max-total-tokens 41943040 \
    --execute > "$queue/valid.log" 2>&1
if [[ "${CAPO_SKIP_SELF_VERIFY:-0}" != "1" ]]; then
    envs/main/bin/python -m inverser run-agentic-baseline "${common[@]}" --method self_verify \
        --output "data/research/$run_id-self-verify" --max-total-tokens 268435456 \
        --execute > "$queue/self-verify.log" 2>&1
fi
test -f "data/research/$run_id-valid/summary.json"

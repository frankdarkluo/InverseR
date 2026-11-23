#!/usr/bin/env bash
# Evaluate every new SFT adapter checkpoint on the valid T4 prefix, one at a time on one GPU.
# Usage: scripts/rolling_t4_eval.sh GPU_ID RUN_DIR [RUN_DIR ...]   (stop: touch artifacts/sft-t4/STOP)
# Each run dir holds checkpoint-N/ (and checkpoint/) adapters; results go to artifacts/sft-t4/rolling.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=$1; shift
root="$PWD" base="${BASE_MODEL:-/data/v-guoqingluo/models/Qwen3.5-9B}"
count="${T4_COUNT:-400}" out="$root/artifacts/sft-t4" port=$((18010 + gpu)) url="http://127.0.0.1:$((18010 + gpu))"
tasks="${TASKS:-T4}"  # T4 (default) or forward; forward rows go to rolling_forward.tsv
tag=t4 tsv="$out/rolling.tsv"
[[ $tasks == forward ]] && tag=fwd tsv="$out/rolling_forward.tsv"
mkdir -p "$out"
[[ -f $out/rolling.tsv ]] || printf 'run\tcheckpoint\tn\tsucc_1r\tmedian_miss_r\twithin_5r\tinvalid\n' > "$out/rolling.tsv"
[[ -f $out/rolling_forward.tsv ]] || printf 'run\tcheckpoint\tn\tT1\tT2_set\tT3_1r\tT3_3r\tS_total\tvalid\n' > "$out/rolling_forward.tsv"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="$root/cache/vllm" HF_HOME="$root/cache/huggingface"
export XDG_CACHE_HOME="$root/cache" TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor" PYTHONPATH="$root/src"
export TMPDIR="${TMPDIR:-/data/v-guoqingluo/tmp}"

evaluate() {  # $1 run dir, $2 checkpoint dir (adapter or merged model)
    local run ck name merged eval_dir server
    run=$(basename "$1") ck=$(basename "$2") name="$run-$ck" eval_dir="data/research/${tag}val-$run-$ck"
    merged="data/research/.rolling-merged/$name"
    if [[ -f $2/adapter_model.safetensors ]]; then
        rm -rf "$merged"
        if ! envs/main/bin/python -m inverser merge-capo-adapter --model-path "$base" --adapter "$2" \
            --output "$merged" --device cuda > "$out/$name.merge.log" 2>&1; then
            printf '%s\t%s\tmerge_failed\n' "$run" "$ck" >> "$tsv"; return 0
        fi
    else
        merged=$2  # already a merged model directory (baselines)
    fi
    envs/main/bin/vllm serve "$merged" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
        --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 \
        --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching --reasoning-parser qwen3 \
        --enforce-eager > "$out/$name.server.log" 2>&1 &
    server=$!
    for _ in $(seq 60); do curl -fsS --max-time 2 "$url/health" >/dev/null 2>&1 && break; sleep 5; done
    envs/main/bin/python -m inverser run-vlm-eval --split data/research/inverser/valid --count "$count" \
        --tasks "$tasks" --model "$name" --revision "sha256:$( (sha256sum "$merged/manifest.json" 2>/dev/null || sha256sum "$merged/config.json") | cut -c1-64)" \
        --local-vllm-url "$url/v1" --thinking --thinking-token-budget 1024 --max-output-tokens 2048 \
        --workers 16 --output "$eval_dir" --max-total-tokens 16777216 --execute > "$out/$name.eval.log" 2>&1 || true
    kill -TERM "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true
    [[ $merged == data/research/.rolling-merged/* ]] && rm -rf "$merged"
    envs/main/bin/python - "$eval_dir" "$run" "$ck" "$tasks" >> "$tsv" <<'PY'
import glob, json, statistics, sys
d, run, ck, task = sys.argv[1:]
rows = [json.load(open(f))["payload"] for f in glob.glob(f"{d}/jobs/*.json")]
rows = [p for p in rows if p.get("task") == task]
if task == "forward":
    keys = ("T1", "T2_cushion_set", "T3_1r", "T3_3r", "S_total")
    scores = [p.get("scores") or {} for p in rows]
    means = [sum(float(s.get(k, 0)) for s in scores) / len(rows) if rows else float("nan") for k in keys]
    valid = sum(bool(p.get("valid")) for p in rows) / len(rows) if rows else float("nan")
    print("\t".join([run, ck, str(len(rows))] + [f"{m:.4f}" for m in means] + [f"{valid:.4f}"]))
    raise SystemExit
loss = [p["attempts"][0]["loss_radii"] for p in rows if p.get("attempts") and p["attempts"][0].get("loss_radii") is not None]
n = len(rows)
succ = sum(bool(p.get("success")) for p in rows) / n if n else float("nan")
med = statistics.median(loss) if loss else float("nan")
w5 = sum(x <= 5 for x in loss) / n if n else float("nan")
print(f"{run}\t{ck}\t{n}\t{succ:.4f}\t{med:.2f}\t{w5:.4f}\t{(n - len(loss)) / n if n else float('nan'):.4f}")
PY
}

while [[ ! -e $out/STOP ]]; do
    for run in "$@"; do
        for ck in "$run"/checkpoint-* "$run"/checkpoint "$run"; do
            [[ -d $ck ]] || continue
            [[ -f $ck/adapter_model.safetensors ]] \
                || { [[ $ck == "$run" ]] && compgen -G "$ck/model*.safetensors" >/dev/null; } || continue
            # Skip files still being written or synced.
            [[ -z $(find "$ck" -maxdepth 1 -newermt '-3 minutes' -print -quit) ]] || continue
            grep -q -P "^$(basename "$run")\t$(basename "$ck")\t" "$tsv" && continue
            evaluate "$run" "$ck"
        done
    done
    sleep 300
done

#!/usr/bin/env bash
# 判断モデルベンチマーク (Issue #18)
#
# SSH 先 (RTX 5090 / 32GB VRAM) で 1 モデルを vLLM で起動し、
# eval ハーネス (#5) でメトリクスと安定性を取って JSON に書き出す。
#
# 使い方:
#   # 単発実行（Qwen3.6-27B AWQ）
#   MODEL="Qwen/Qwen3.6-27B-Instruct-AWQ" \
#   SERVED_NAME="qwen3.6-27b-awq" \
#   bash scripts/run_model_benchmark.sh
#
#   # 全候補を順に回す
#   bash scripts/run_model_benchmark.sh --all
#
# 出力:
#   reports/model_benchmarks/{served_name}/{timestamp}.json   # make eval の結果
#   reports/model_benchmarks/{served_name}/{timestamp}_stability.json
#   reports/model_benchmarks/{served_name}/{timestamp}_latency.json
#
# 前提:
#   - SSH 先で `uv pip install vllm` 済み
#   - リポジトリが clone されている（カレントディレクトリがリポジトリ root）
#   - `data/annotations/gold/v1/` にゴールデンアノテが置かれている (#6 完了後)
#     アノテ未整備でも `data/sample_meetings/sample_meeting_01.json` で latency / stability は取れる

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# venv 自動有効化（uv sync は backend/.venv または ./.venv を作るので両方探す）
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for venv_path in "$REPO_ROOT/.venv" "$REPO_ROOT/backend/.venv"; do
    if [[ -f "$venv_path/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$venv_path/bin/activate"
      echo "Activated venv: $venv_path"
      break
    fi
  done
fi

# Python 実行コマンドは PYTHON で上書き可能。
# 既定は `python`（venv 有効化後はそちらが使われる）。venv 未使用なら PYTHON=python3 推奨。
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
    echo "WARN: python が無いので python3 を使います"
  else
    echo "ERROR: python / python3 が見つかりません" >&2
    exit 1
  fi
fi

# vLLM が import できるか事前チェック
if ! "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON で vllm が import できません。" >&2
  echo "       \`uv pip install vllm\` 済みの venv を有効化するか、PYTHON= を指定してください。" >&2
  exit 1
fi

# Blackwell (sm_120 / RTX 5090) では FlashInfer の JIT が "requires sm75 or higher"
# (実態は sm_120 認識失敗) で落ちる。FlashAttention に倒す。
# uninstall flashinfer-python が確実だが、env でも切り替えを試みる。
# 既定値: FLASH_ATTN。Blackwell + bnb + bf16 のいずれでも v1 エンジンで動く想定。
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
# JIT が起きる場合の CUDA arch (Blackwell)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# FlashInfer が import 可能だと v1 エンジンが優先選択しがちなので、
# pip uninstall flashinfer-python しておくと確実 (本スクリプトは uninstall は行わない)。
if "$PYTHON" -c "import flashinfer" >/dev/null 2>&1; then
  echo "WARN: flashinfer-python が import 可能です。Blackwell では未対応で落ちることがあります。"
  echo "      回避するには:  uv pip uninstall flashinfer-python"
fi

DATASET="${DATASET:-data/annotations/gold/v1}"
SAMPLE="${SAMPLE:-data/sample_meetings/sample_meeting_01.json}"
N_STABILITY="${N_STABILITY:-5}"
N_LATENCY="${N_LATENCY:-100}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# 32GB VRAM 環境では既定 8K でも 27B/32B クラスは KV cache で OOM するため 4K に絞る
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# KV cache を fp8 にして容量を半減（精度劣化は軽微）
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# 同時並列実行数（既定 256 だと KV cache を食いすぎる）
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
LOG_DIR="${LOG_DIR:-/tmp/meeting-score-vllm}"
mkdir -p "$LOG_DIR"

# --------------------------------------------------------------------------
# 候補モデル一覧（--all 時に順に回す）
# 形式: "HF_ID|SERVED_NAME|EXTRA_VLLM_ARGS"
#
# EXTRA に --enforce-eager を付けるとモデル本体側のメモリ削減。
# 27B BnB は CUDA graph のための余分なバッファすら確保できないので必須。
# --------------------------------------------------------------------------
CANDIDATES=(
  # 第一採用候補: Qwen3.6-35B-A3B (NVFP4 / compressed-tensors)
  # MoE で総 35B / アクティブ 3B、Blackwell ネイティブ FP4 量子化。
  # unsloth が公式 NVFP4 を公開しており、`config.json` の quantization_config は
  # `compressed-tensors` フォーマット (vLLM 0.20+ で対応)。
  "unsloth/Qwen3.6-35B-A3B-NVFP4|qwen3.6-35b-nvfp4|--quantization compressed-tensors --enforce-eager"
  # 控え候補: Qwen2.5-32B-Instruct-AWQ
  # 公式 AWQ + Marlin で 32GB に余裕で乗り、レイテンシ最速級
  "Qwen/Qwen2.5-32B-Instruct-AWQ|qwen2.5-32b-awq|--quantization awq_marlin --dtype auto --enforce-eager"
  # 比較対象: Qwen3.6-27B (BnB) / Qwen3-14B (BnB)
  # 公式量子化が無いため BnB オンザフライ。BnB はカーネル最適化が AWQ 比で遅い
  "Qwen/Qwen3.6-27B|qwen3.6-27b-bnb|--quantization bitsandbytes --dtype auto --enforce-eager"
  "Qwen/Qwen3-14B|qwen3-14b-bnb|--quantization bitsandbytes --dtype auto --enforce-eager"
  # 14B 別系統対照: Phi-4 (MIT)
  "microsoft/phi-4|phi-4-14b-bnb|--quantization bitsandbytes --dtype auto --enforce-eager"
  # 8B 日本語特化対照: Swallow
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3|swallow-3.1-8b-bf16|--dtype bfloat16"
)

# --------------------------------------------------------------------------
# 関数: 1 モデルを起動 → eval → stability → latency → 終了
# --------------------------------------------------------------------------
kill_process_tree() {
  local signal="$1"
  local root_pid="$2"
  local child_pid

  while IFS= read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    kill_process_tree "$signal" "$child_pid"
  done < <(pgrep -P "$root_pid" 2>/dev/null || true)

  kill "-$signal" "$root_pid" 2>/dev/null || true
}

benchmark_one() {
  local hf_id="$1"
  local served="$2"
  local extra="$3"
  local out_dir="reports/model_benchmarks/${served}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$out_dir"

  echo "==================================================================="
  echo "  Benchmark : ${served}"
  echo "  HF model  : ${hf_id}"
  echo "  Extra args: ${extra}"
  echo "  Output    : ${out_dir}/${ts}_*.json"
  echo "==================================================================="

  # vLLM をバックグラウンドで起動
  # vLLM 0.20+ では guided-decoding-backend は廃止され、xgrammar が既定。
  # 旧版の場合は GUIDED_BACKEND 環境変数で指定可。
  local vllm_log="${LOG_DIR}/${served}_${ts}.log"
  local guided_args=""
  if [[ -n "${GUIDED_BACKEND:-}" ]]; then
    guided_args="--guided-decoding-backend ${GUIDED_BACKEND}"
  fi
  local vllm_process_group=0
  if command -v setsid >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$hf_id" \
        --served-model-name "$served" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        $guided_args \
        $extra > "$vllm_log" 2>&1 &
    vllm_process_group=1
  else
    # shellcheck disable=SC2086
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$hf_id" \
        --served-model-name "$served" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        $guided_args \
        $extra > "$vllm_log" 2>&1 &
  fi
  local vllm_pid=$!
  terminate_vllm() {
    local signal="$1"

    if [[ "$vllm_process_group" -eq 1 ]]; then
      kill "-$signal" -- "-$vllm_pid" 2>/dev/null || true
    else
      kill_process_tree "$signal" "$vllm_pid"
    fi
  }
  vllm_is_running() {
    if [[ "$vllm_process_group" -eq 1 ]]; then
      kill -0 -- "-$vllm_pid" 2>/dev/null
    else
      kill -0 "$vllm_pid" 2>/dev/null
    fi
  }
  trap 'terminate_vllm TERM' EXIT

  # ヘルスチェックでロード完了を待つ
  # 既定 1800 秒 = 30 分。27B BnB 等の初回ロードは長い。
  local max_wait_sec="${VLLM_READY_TIMEOUT:-1800}"
  local iters=$(( max_wait_sec / 10 ))
  local ready=0
  echo "Waiting for vLLM to load model (timeout ${max_wait_sec}s)..."
  for i in $(seq 1 "$iters"); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "  ready after ~$((i * 10))s"
      ready=1
      break
    fi
    sleep 10
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "ERROR: vLLM died before becoming ready. See $vllm_log" >&2
      exit 1
    fi
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: vLLM did not become ready within ${max_wait_sec}s. See $vllm_log" >&2
    exit 1
  fi

  # eval CLI に渡す共通フラグ。--backend local + --endpoint で vLLM サーバを叩く
  local endpoint="http://127.0.0.1:${PORT}/v1"
  local cli_common=(--backend local --endpoint "$endpoint" --model "$served")

  # --- 1) ベースライン評価（アノテがあれば） ---
  if [[ -d "$DATASET" && -f "$DATASET/pairs.jsonl" ]]; then
    echo "[1/3] make eval"
    (cd backend && "$PYTHON" -m evals.cli "${cli_common[@]}" run \
        --dataset "../${DATASET}" \
        --out "../${out_dir}/${ts}.json") || echo "WARN: eval failed"
  else
    echo "[1/3] eval をスキップ (アノテ未整備: $DATASET)"
  fi

  # --- 2) 安定性（N 回採点して軸別 SD） ---
  echo "[2/3] stability N=${N_STABILITY}"
  (cd backend && "$PYTHON" -m evals.cli "${cli_common[@]}" stability \
      --meeting "../${SAMPLE}" \
      --n "$N_STABILITY" \
      --out "../${out_dir}/${ts}_stability.json") || echo "WARN: stability failed"

  # --- 3) レイテンシ（同一発言を N 回。同期呼び出しで p50/p95） ---
  echo "[3/3] latency N=${N_LATENCY}"
  "$PYTHON" scripts/measure_latency.py \
      --endpoint "http://127.0.0.1:${PORT}/v1" \
      --model "$served" \
      --sample "$SAMPLE" \
      --n "$N_LATENCY" \
      --out "${out_dir}/${ts}_latency.json" || echo "WARN: latency failed"

  # vLLM 終了とメモリ解放 (次モデルへの residual を避ける)
  # 1) SIGTERM、起動時のプロセスグループまたは PID 配下ごと
  terminate_vllm TERM

  # 2) 最大 30 秒待って、まだ生きてたら SIGKILL
  for i in $(seq 1 30); do
    vllm_is_running || break
    sleep 1
  done
  if vllm_is_running; then
    echo "  vLLM が SIGTERM で終わらないので SIGKILL"
    terminate_vllm KILL
  fi
  wait "$vllm_pid" 2>/dev/null || true

  # 3) GPU メモリ解放を確認 (5090 32GB なら使用量が ~500MB 以下に落ちる想定)
  sleep 5
  if command -v nvidia-smi >/dev/null 2>&1; then
    local used_mb
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "  GPU memory after shutdown: ${used_mb} MiB used"
  fi

  trap - EXIT
  echo "Done: ${served}"
}

# --------------------------------------------------------------------------
# エントリポイント
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
  for entry in "${CANDIDATES[@]}"; do
    IFS='|' read -r hf_id served extra <<< "$entry"
    benchmark_one "$hf_id" "$served" "$extra"
  done
else
  : "${MODEL:?MODEL を指定してください（例: MODEL=Qwen/Qwen3.6-27B）}"
  : "${SERVED_NAME:?SERVED_NAME を指定してください（例: SERVED_NAME=qwen3.6-27b-bnb）}"
  EXTRA="${EXTRA:---dtype auto}"
  benchmark_one "$MODEL" "$SERVED_NAME" "$EXTRA"
fi

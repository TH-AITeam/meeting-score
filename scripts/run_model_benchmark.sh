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

DATASET="${DATASET:-data/annotations/gold/v1}"
SAMPLE="${SAMPLE:-data/sample_meetings/sample_meeting_01.json}"
N_STABILITY="${N_STABILITY:-5}"
N_LATENCY="${N_LATENCY:-100}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOG_DIR="${LOG_DIR:-/tmp/meeting-score-vllm}"
mkdir -p "$LOG_DIR"

# --------------------------------------------------------------------------
# 候補モデル一覧（--all 時に順に回す）
# 形式: "HF_ID|SERVED_NAME|EXTRA_VLLM_ARGS"
# --------------------------------------------------------------------------
CANDIDATES=(
  "Qwen/Qwen3.6-27B-Instruct-AWQ|qwen3.6-27b-awq|--quantization awq_marlin --dtype auto"
  "Qwen/Qwen3-14B-Instruct|qwen3-14b-bf16|--dtype bfloat16"
  "meta-llama/Llama-3.3-8B-Instruct|llama-3.3-8b-bf16|--dtype bfloat16"
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3|swallow-3.1-8b-bf16|--dtype bfloat16"
  "microsoft/phi-4|phi-4-14b-bf16|--dtype bfloat16"
)

# --------------------------------------------------------------------------
# 関数: 1 モデルを起動 → eval → stability → latency → 終了
# --------------------------------------------------------------------------
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
  # shellcheck disable=SC2086
  "$PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$hf_id" \
      --served-model-name "$served" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      $guided_args \
      $extra > "$vllm_log" 2>&1 &
  local vllm_pid=$!
  trap 'kill $vllm_pid 2>/dev/null || true' EXIT

  # ヘルスチェックでロード完了を待つ（最大 600 秒）
  echo "Waiting for vLLM to load model..."
  for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "  ready after ~$((i * 10))s"
      break
    fi
    sleep 10
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "ERROR: vLLM died before becoming ready. See $vllm_log" >&2
      exit 1
    fi
  done

  # config.yaml は環境変数で上書きするか、テスト用 config を流す
  export MEETING_SCORE_LLM_BACKEND="local"
  export MEETING_SCORE_LLM_ENDPOINT="http://127.0.0.1:${PORT}/v1"
  export MEETING_SCORE_LLM_MODEL="$served"

  # --- 1) ベースライン評価（アノテがあれば） ---
  if [[ -d "$DATASET" && -f "$DATASET/pairs.jsonl" ]]; then
    echo "[1/3] make eval"
    (cd backend && "$PYTHON" -m evals.cli run \
        --dataset "../${DATASET}" \
        --model "$served" \
        --out "../${out_dir}/${ts}.json") || echo "WARN: eval failed"
  else
    echo "[1/3] eval をスキップ (アノテ未整備: $DATASET)"
  fi

  # --- 2) 安定性（N 回採点して軸別 SD） ---
  echo "[2/3] stability N=${N_STABILITY}"
  (cd backend && "$PYTHON" -m evals.cli stability \
      --meeting "../${SAMPLE}" \
      --n "$N_STABILITY" \
      --model "$served" \
      --out "../${out_dir}/${ts}_stability.json") || echo "WARN: stability failed"

  # --- 3) レイテンシ（同一発言を N 回。同期呼び出しで p50/p95） ---
  echo "[3/3] latency N=${N_LATENCY}"
  "$PYTHON" scripts/measure_latency.py \
      --endpoint "http://127.0.0.1:${PORT}/v1" \
      --model "$served" \
      --sample "$SAMPLE" \
      --n "$N_LATENCY" \
      --out "${out_dir}/${ts}_latency.json" || echo "WARN: latency failed"

  # vLLM 終了
  kill "$vllm_pid" 2>/dev/null || true
  wait "$vllm_pid" 2>/dev/null || true
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
  : "${MODEL:?MODEL を指定してください（例: MODEL=Qwen/Qwen3.6-27B-Instruct-AWQ）}"
  : "${SERVED_NAME:?SERVED_NAME を指定してください（例: SERVED_NAME=qwen3.6-27b-awq）}"
  EXTRA="${EXTRA:---dtype auto}"
  benchmark_one "$MODEL" "$SERVED_NAME" "$EXTRA"
fi

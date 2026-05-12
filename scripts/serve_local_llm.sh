#!/usr/bin/env bash
# vLLM OpenAI 互換サーバ起動スクリプト (Issue #12)
#
# 採用モデル名は Issue #18 (判断モデル選定) で決定後、
# 環境変数 MODEL を上書きする想定。
# 比較ベンチマークを回したい場合は scripts/run_model_benchmark.sh を使う。
#
# 使い方:
#   # 既定モデルで起動
#   bash scripts/serve_local_llm.sh
#
#   # モデル/ポート上書き
#   MODEL="Qwen/Qwen2.5-7B-Instruct" PORT=8000 bash scripts/serve_local_llm.sh
#
#   # GPU メモリ割合を変える
#   GPU_MEM_UTIL=0.85 bash scripts/serve_local_llm.sh
#
# 動作前提:
#   uv pip install vllm
#   または: pip install vllm
#   GPU: A100 を想定（8B クラスなら fp16 で 1GPU で動く）
#
# 応答形式 JSON 強制は OpenAI 互換 response_format=json_schema に対し、
# vLLM の --guided-decoding-backend xgrammar が解釈する。

set -euo pipefail

# venv 自動有効化
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for venv_path in "$REPO_ROOT/.venv" "$REPO_ROOT/backend/.venv"; do
    if [[ -f "$venv_path/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$venv_path/bin/activate"
      break
    fi
  done
fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="python3"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GUIDED_BACKEND="${GUIDED_BACKEND:-xgrammar}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"

echo "============================================"
echo "  vLLM OpenAI 互換サーバ起動"
echo "============================================"
echo "  MODEL         : ${MODEL}"
echo "  HOST:PORT     : ${HOST}:${PORT}"
echo "  GPU_MEM_UTIL  : ${GPU_MEM_UTIL}"
echo "  MAX_MODEL_LEN : ${MAX_MODEL_LEN}"
echo "  GUIDED        : ${GUIDED_BACKEND}"
echo "  Served as     : ${SERVED_MODEL_NAME}"
echo "============================================"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --guided-decoding-backend "${GUIDED_BACKEND}" \
    --served-model-name "${SERVED_MODEL_NAME}"

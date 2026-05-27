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
#   MODEL="Qwen/Qwen2.5-7B-Instruct" PORT=8001 bash scripts/serve_local_llm.sh
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
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# vLLM 0.20+ では --guided-decoding-backend は削除され xgrammar が既定。
# 旧 vLLM (<0.20) を使う場合のみ GUIDED_BACKEND を指定する。
GUIDED_BACKEND="${GUIDED_BACKEND:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"

# 組織別 LoRA マルチアダプタ配信 (Issue #83)。
#   ENABLE_LORA=1 で --enable-lora を付与。MAX_LORAS は同時ロード上限
#   (GPU 80GB なら 8 程度が無難)。LORA_MODULES は起動時登録するアダプタ
#   ("org_001=adapters/org_001/v3 org_002=...")。未指定なら adapters/registry.yaml
#   の active_version から自動生成する。未ロード組織は初回リクエスト時に
#   vLLM 拡張 API (/v1/load_lora_adapter) でロードし、上限超過は LRU で押し出す。
ENABLE_LORA="${ENABLE_LORA:-0}"
MAX_LORAS="${MAX_LORAS:-8}"
LORA_MODULES="${LORA_MODULES:-}"
if [[ "${ENABLE_LORA}" == "1" && -z "${LORA_MODULES}" ]]; then
  LORA_MODULES="$(PYTHONPATH="$REPO_ROOT/backend" "$PYTHON" -c "
from app.evaluators.adapter_resolver import AdapterResolver
print(' '.join(AdapterResolver(base_model='${MODEL}').lora_modules()))
" 2>/dev/null || true)"
fi

echo "============================================"
echo "  vLLM OpenAI 互換サーバ起動"
echo "============================================"
echo "  MODEL         : ${MODEL}"
echo "  HOST:PORT     : ${HOST}:${PORT}"
echo "  GPU_MEM_UTIL  : ${GPU_MEM_UTIL}"
echo "  MAX_MODEL_LEN : ${MAX_MODEL_LEN}"
echo "  GUIDED        : ${GUIDED_BACKEND:-default(xgrammar)}"
echo "  Served as     : ${SERVED_MODEL_NAME}"
echo "  ENABLE_LORA   : ${ENABLE_LORA} (max=${MAX_LORAS})"
echo "  LORA_MODULES  : ${LORA_MODULES:-(なし)}"
echo "============================================"

GUIDED_ARGS=""
if [[ -n "${GUIDED_BACKEND}" ]]; then
  GUIDED_ARGS="--guided-decoding-backend ${GUIDED_BACKEND}"
fi
LORA_ARGS=""
if [[ "${ENABLE_LORA}" == "1" ]]; then
  LORA_ARGS="--enable-lora --max-loras ${MAX_LORAS}"
  if [[ -n "${LORA_MODULES}" ]]; then
    LORA_ARGS="${LORA_ARGS} --lora-modules ${LORA_MODULES}"
  fi
fi
# shellcheck disable=SC2086
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${GUIDED_ARGS} \
    ${LORA_ARGS} \
    --served-model-name "${SERVED_MODEL_NAME}"

#!/usr/bin/env bash
# 音声処理モデルベンチマーク (Issue #19)
#
# data/eval_audio/ に並んだ評価音声 (最低 3 本、自作録音または合成音声) に
# 対し、ADR 0002 で挙げた ASR / Diarization 候補それぞれを動かして
# CER / RTF / DER / 固有名詞認識率を計測する。
#
# 評価音声の準備手順は data/eval_audio/README.md を参照。
#
# 使い方:
#   # 全候補 (ASR × Diar の組合せ) を順に回す
#   bash scripts/run_audio_benchmark.sh --all
#
#   # 単体 ASR だけ
#   ASR=whisperx-large-v3 bash scripts/run_audio_benchmark.sh asr
#
#   # 単体 Diar だけ
#   DIAR=pyannote-3.1 bash scripts/run_audio_benchmark.sh diar
#
# 前提:
#   - SSH 先 (RTX 5090 / 32GB) で uv 環境セットアップ済み
#   - `uv sync --extra audio` で whisperx / pyannote-audio / librosa を導入済み
#   - HUGGINGFACE_HUB_TOKEN を export 済み (pyannote の gated repo アクセス用)
#   - data/eval_audio/{meeting_01,02,03}/{audio.wav, reference.txt, speakers.rttm}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# venv 自動有効化
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
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python3"
fi

# 依存チェック
for mod in whisperx pyannote.audio librosa; do
  if ! "$PYTHON" -c "import $mod" >/dev/null 2>&1; then
    echo "ERROR: $mod が import できません。'uv sync --extra audio' で導入してください。" >&2
    exit 1
  fi
done

# HF token チェック (pyannote の gated repo)
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HUGGINGFACE_HUB_TOKEN が未設定です。pyannote/speaker-diarization-3.1 は gated repo です。" >&2
  echo "       'export HUGGINGFACE_HUB_TOKEN=hf_xxx' で設定してください。" >&2
  exit 1
fi

# 評価音声チェック
EVAL_DIR="${EVAL_DIR:-data/eval_audio}"
if [[ ! -d "$EVAL_DIR" ]]; then
  echo "ERROR: 評価音声ディレクトリ $EVAL_DIR が見つかりません。" >&2
  echo "       data/eval_audio/README.md の手順で準備してください。" >&2
  exit 1
fi
MEETING_COUNT=$(find "$EVAL_DIR" -maxdepth 1 -mindepth 1 -type d ! -name "_*" | wc -l | tr -d ' ')
if [[ "$MEETING_COUNT" -lt 3 ]]; then
  echo "WARN: 評価音声が $MEETING_COUNT 本しかありません (Issue #19 仕様は最低 3 本)。" >&2
fi

REPORT_DIR="${REPORT_DIR:-reports/audio_benchmarks}"
mkdir -p "$REPORT_DIR"

# 候補リスト (ADR 0002)
ASR_CANDIDATES=(
  "whisperx-large-v3|openai/whisper-large-v3"
  "whisperx-kotoba-v2|kotoba-tech/kotoba-whisper-v2.0"
)
DIAR_CANDIDATES=(
  "pyannote-3.1|pyannote/speaker-diarization-3.1"
)

run_asr_one() {
  local label="$1"
  local hf_id="$2"
  echo "========================================="
  echo "  ASR : $label  ($hf_id)"
  echo "========================================="
  "$PYTHON" scripts/measure_asr_metrics.py \
    --asr-id "$label" \
    --whisper-model "$hf_id" \
    --eval-dir "$EVAL_DIR" \
    --out "$REPORT_DIR/${label}/asr.json"
}

run_diar_one() {
  local label="$1"
  local hf_id="$2"
  echo "========================================="
  echo "  Diar: $label  ($hf_id)"
  echo "========================================="
  "$PYTHON" scripts/measure_diar_metrics.py \
    --diar-id "$label" \
    --diar-model "$hf_id" \
    --eval-dir "$EVAL_DIR" \
    --out "$REPORT_DIR/${label}/diar.json"
}

case "${1:-}" in
  --all)
    for entry in "${ASR_CANDIDATES[@]}"; do
      IFS='|' read -r label hf_id <<< "$entry"
      run_asr_one "$label" "$hf_id"
    done
    for entry in "${DIAR_CANDIDATES[@]}"; do
      IFS='|' read -r label hf_id <<< "$entry"
      run_diar_one "$label" "$hf_id"
    done
    ;;
  asr)
    : "${ASR:?ASR を指定してください (例: ASR=whisperx-large-v3)}"
    for entry in "${ASR_CANDIDATES[@]}"; do
      IFS='|' read -r label hf_id <<< "$entry"
      if [[ "$label" == "$ASR" ]]; then run_asr_one "$label" "$hf_id"; fi
    done
    ;;
  diar)
    : "${DIAR:?DIAR を指定してください (例: DIAR=pyannote-3.1)}"
    for entry in "${DIAR_CANDIDATES[@]}"; do
      IFS='|' read -r label hf_id <<< "$entry"
      if [[ "$label" == "$DIAR" ]]; then run_diar_one "$label" "$hf_id"; fi
    done
    ;;
  *)
    echo "Usage: bash scripts/run_audio_benchmark.sh [--all | asr | diar]" >&2
    exit 1
    ;;
esac

echo "Done. 結果: $REPORT_DIR/*/{asr,diar}.json"

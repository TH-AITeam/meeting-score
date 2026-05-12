"""ASR の CER / RTF / 固有名詞認識率を計測する (Issue #19)。

評価データの配置 (data/eval_audio/README.md 参照):

    data/eval_audio/
      meeting_01/
        audio.wav            # 録音 or 合成 (最低 15 分)
        reference.txt        # 正解文字起こし (全文)
        named_entities.txt   # 固有名詞 (1 行 1 語、任意)
      meeting_02/...
      meeting_03/...

実行:
    python scripts/measure_asr_metrics.py \\
        --asr-id whisperx-large-v3 \\
        --whisper-model openai/whisper-large-v3 \\
        --eval-dir data/eval_audio \\
        --out reports/audio_benchmarks/whisperx-large-v3/asr.json

本スクリプトは WhisperX のロード + 推論実装が **本実装 (Issue #11)** 完了
後に動く前提で書いている。Issue #19 時点ではスケルトンとして、CLI 形と
出力 JSON フォーマットを確定させる。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """文字単位の編集距離 / 参照長。"""
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    # Wagner-Fischer DP
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        curr = [i]
        for j, hc in enumerate(hyp, start=1):
            sub = prev[j - 1] + (0 if rc == hc else 1)
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            curr.append(min(sub, ins, dele))
        prev = curr
    return prev[-1] / len(ref)


def _named_entity_recall(hypothesis: str, entities: list[str]) -> float:
    if not entities:
        return float("nan")
    hits = sum(1 for e in entities if e and e in hypothesis)
    return hits / len(entities)


def _transcribe_with_whisperx(audio_path: Path, model_name: str) -> tuple[str, float]:
    """WhisperX で 1 ファイルを書き起こす (Issue #11 本実装)。

    `app.asr.WhisperXTranscriber` を再利用する。

    Returns
    -------
    (text, audio_sec)
        text: 全 Word を空文字結合した書き起こし全文
        audio_sec: 音声の長さ (RTF 計算に使う)
    """
    # 本スクリプトは scripts/ 配下なので、backend/ を sys.path に追加して app をロードする
    repo_root = Path(__file__).resolve().parent.parent
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        import soundfile as sf
    except ImportError as e:
        msg = "soundfile 未導入。`uv sync --extra audio` を実行してください。"
        raise RuntimeError(msg) from e

    from app.asr.whisperx_transcriber import WhisperXConfig, WhisperXTranscriber

    # 音声長を先に取る (RTF 計算用)
    info = sf.info(str(audio_path))
    audio_sec = float(info.frames) / float(info.samplerate)

    transcriber = WhisperXTranscriber(
        WhisperXConfig(
            model_name=model_name,
            device="cuda",
            compute_type="float16",
            language="ja",
        )
    )
    words = transcriber.transcribe(audio_path)
    text = "".join(w.text for w in words)
    return text, audio_sec


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR メトリクス計測")
    parser.add_argument("--asr-id", required=True, help="出力 JSON に書く ASR ラベル")
    parser.add_argument("--whisper-model", required=True, help="HF モデル ID")
    parser.add_argument("--eval-dir", required=True, help="評価音声のルート")
    parser.add_argument("--out", required=True, help="JSON 出力先")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meetings = sorted(
        d for d in eval_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    if not meetings:
        msg = f"評価会議が {eval_dir} に見つかりません。"
        raise SystemExit(msg)

    per_meeting = []
    total_wall = 0.0
    total_audio_sec = 0.0
    for m_dir in meetings:
        audio = m_dir / "audio.wav"
        if not audio.exists():
            logger.warning("audio.wav が無いのでスキップ: %s", m_dir)
            continue
        ref_text = (m_dir / "reference.txt").read_text(encoding="utf-8").strip()
        entities = _read_lines(m_dir / "named_entities.txt")

        start = time.perf_counter()
        try:
            hyp, audio_sec = _transcribe_with_whisperx(audio, args.whisper_model)
        except Exception as e:  # noqa: BLE001 - WhisperX 失敗時は当該 meeting を skip
            logger.exception("transcribe failed: %s", m_dir.name)
            per_meeting.append(
                {
                    "meeting": m_dir.name,
                    "status": "failed",
                    "error": str(e),
                }
            )
            continue
        wall_sec = time.perf_counter() - start
        cer = _char_error_rate(ref_text, hyp)
        ne_recall = _named_entity_recall(hyp, entities)
        total_wall += wall_sec
        total_audio_sec += audio_sec
        per_meeting.append(
            {
                "meeting": m_dir.name,
                "cer": cer,
                "named_entity_recall": ne_recall,
                "audio_sec": audio_sec,
                "wall_sec": wall_sec,
            }
        )

    rtf = total_wall / total_audio_sec if total_audio_sec > 0 else None
    payload = {
        "asr_id": args.asr_id,
        "model": args.whisper_model,
        "per_meeting": per_meeting,
        "aggregate": {
            "macro_cer": _macro([m.get("cer") for m in per_meeting]),
            "macro_named_entity_recall": _macro(
                [m.get("named_entity_recall") for m in per_meeting]
            ),
            "rtf": rtf,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _macro(values: list[float | None]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not _is_nan(v)]
    return sum(nums) / len(nums) if nums else None


def _is_nan(v: float) -> bool:
    try:
        return v != v  # noqa: PLR0124
    except TypeError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

"""Diarization の DER / overlap accuracy を計測する (Issue #19)。

評価データ (data/eval_audio/README.md 参照):
    data/eval_audio/meeting_XX/
      audio.wav
      speakers.rttm    # 正解 RTTM (pyannote 標準フォーマット)

実行:
    python scripts/measure_diar_metrics.py \\
        --diar-id pyannote-3.1 \\
        --diar-model pyannote/speaker-diarization-3.1 \\
        --eval-dir data/eval_audio \\
        --out reports/audio_benchmarks/pyannote-3.1/diar.json

本実装 (pyannote パイプライン呼び出し) は Issue #11 で完成させる。
本 PR はスケルトン (CLI 形 + JSON 出力フォーマット) のみ。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _diarize_with_pyannote(audio_path: Path, model_name: str) -> Path:
    """pyannote で 1 ファイルを diarize し、RTTM ファイルパスを返す。

    Issue #11 で本実装。本 PR ではスケルトン。
    """
    try:
        from pyannote.audio import Pipeline  # noqa: F401
    except ImportError as e:
        msg = "pyannote-audio 未導入。`uv sync --extra audio` を実行してください。"
        raise RuntimeError(msg) from e
    if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        msg = "HUGGINGFACE_HUB_TOKEN が未設定。pyannote/speaker-diarization-3.1 は gated。"
        raise RuntimeError(msg)
    msg = (
        "_diarize_with_pyannote は Issue #11 で実装予定。"
        "現状は CLI と出力 JSON のフォーマットを確定するためのスケルトン。"
    )
    _ = audio_path, model_name
    raise NotImplementedError(msg)


def _compute_der(reference_rttm: Path, hypothesis_rttm: Path) -> dict[str, float]:
    """`pyannote.metrics` で DER と overlap accuracy を計算する。"""
    try:
        from pyannote.core import Annotation  # noqa: F401
        from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: F401
    except ImportError as e:
        msg = "pyannote.metrics 未導入。`uv sync --extra audio` を実行してください。"
        raise RuntimeError(msg) from e
    msg = "_compute_der は Issue #11 で実装予定。"
    _ = reference_rttm, hypothesis_rttm
    raise NotImplementedError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diarization メトリクス計測")
    parser.add_argument("--diar-id", required=True)
    parser.add_argument("--diar-model", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out", required=True)
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
    for m_dir in meetings:
        audio = m_dir / "audio.wav"
        ref_rttm = m_dir / "speakers.rttm"
        if not audio.exists() or not ref_rttm.exists():
            logger.warning("audio.wav / speakers.rttm 不足: %s", m_dir)
            continue
        try:
            hyp_rttm = _diarize_with_pyannote(audio, args.diar_model)
            metrics = _compute_der(ref_rttm, hyp_rttm)
        except NotImplementedError as e:
            per_meeting.append(
                {
                    "meeting": m_dir.name,
                    "status": "skipped (impl pending in #11)",
                    "skip_reason": str(e),
                }
            )
            continue
        per_meeting.append({"meeting": m_dir.name, **metrics})

    payload = {
        "diar_id": args.diar_id,
        "model": args.diar_model,
        "per_meeting": per_meeting,
        "aggregate": {
            "macro_der": _macro([m.get("der") for m in per_meeting]),
            "macro_overlap_accuracy": _macro(
                [m.get("overlap_accuracy") for m in per_meeting]
            ),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _macro(values: list[float | None]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


if __name__ == "__main__":
    raise SystemExit(main())

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


def _diarize_with_pyannote(
    audio_path: Path,
    model_name: str,
    out_rttm_path: Path,
) -> Path:
    """pyannote で 1 ファイルを diarize し、RTTM 形式で書き出してパスを返す。"""
    # scripts/ から backend/ を sys.path に
    repo_root = Path(__file__).resolve().parent.parent
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        msg = "HUGGINGFACE_HUB_TOKEN が未設定。pyannote/speaker-diarization-3.1 は gated。"
        raise RuntimeError(msg)

    from app.asr.pyannote_diarizer import PyannoteConfig, PyannoteDiarizer

    diarizer = PyannoteDiarizer(
        PyannoteConfig(model_name=model_name, device="cuda")
    )
    turns = diarizer.diarize(audio_path)

    # pyannote 標準 RTTM 形式で書き出す
    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = audio_path.stem
    with out_rttm_path.open("w", encoding="utf-8") as f:
        for t in turns:
            duration = max(0.0, t.end_sec - t.start_sec)
            f.write(
                f"SPEAKER {file_id} 1 {t.start_sec:.3f} {duration:.3f} "
                f"<NA> <NA> {t.speaker} <NA> <NA>\n"
            )
    return out_rttm_path


def _compute_der(
    reference_rttm: Path, hypothesis_rttm: Path
) -> dict[str, float | None]:
    """`pyannote.metrics` で DER と overlap accuracy を計算する。

    NaN や Inf は JSON 標準で invalid のため、`None` に正規化して返す。
    """
    import math

    try:
        from pyannote.database.util import load_rttm
        from pyannote.metrics.detection import DetectionAccuracy
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as e:
        msg = "pyannote.metrics 未導入。`uv sync --extra audio` を実行してください。"
        raise RuntimeError(msg) from e

    ref_anno = next(iter(load_rttm(str(reference_rttm)).values()))
    hyp_anno = next(iter(load_rttm(str(hypothesis_rttm)).values()))

    der_metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    der_raw = float(der_metric(ref_anno, hyp_anno))
    der: float | None = (
        None if (math.isnan(der_raw) or math.isinf(der_raw)) else der_raw
    )

    # overlap detection accuracy: 「ref の overlap 区間」と「hyp の overlap 区間」を比較
    # pyannote の get_overlap() で overlap-only annotation を取り出して比較
    overlap_acc: float | None
    try:
        ref_overlap = ref_anno.get_overlap()
        hyp_overlap = hyp_anno.get_overlap()
        acc_metric = DetectionAccuracy()
        val = float(acc_metric(ref_overlap, hyp_overlap))
        overlap_acc = None if (math.isnan(val) or math.isinf(val)) else val
    except Exception:  # noqa: BLE001 - overlap が無い等のエッジケース
        overlap_acc = None

    return {"der": der, "overlap_accuracy": overlap_acc}


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
            hyp_rttm = out_path.parent / f"{m_dir.name}_hyp.rttm"
            _diarize_with_pyannote(audio, args.diar_model, hyp_rttm)
            metrics = _compute_der(ref_rttm, hyp_rttm)
        except Exception as e:  # noqa: BLE001
            logger.exception("diar failed: %s", m_dir.name)
            per_meeting.append(
                {
                    "meeting": m_dir.name,
                    "status": "failed",
                    "error": str(e),
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
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


def _macro(values: list[float | None]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


if __name__ == "__main__":
    raise SystemExit(main())

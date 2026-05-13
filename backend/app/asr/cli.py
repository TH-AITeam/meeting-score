"""音声入力パイプライン CLI (Issue #11, #68)。

音声ファイル (wav/mp3/m4a/webm/...) または動画ファイル (mp4/mov/mkv/...) を
MeetingInput JSON に一気通貫で変換する。

Issue #68 で動画拡張子サポートを追加。動画入力時はローカル ffmpeg で音声を
一時抽出し、既存パイプラインに渡す (backend API のように動画拒否はしない)。

Examples
--------
    # 音声ファイル → MeetingInput
    python -m app.asr.cli --input meeting.wav --output meeting.json --meeting-id m042

    # 動画ファイル → MeetingInput (Issue #68): ローカル ffmpeg で音声抽出
    python -m app.asr.cli --input meeting.mp4 --output meeting.json --meeting-id m042

    # メタ抽出をスキップ (LLM 呼び出し無し、title/goal は手で埋める想定)
    python -m app.asr.cli --input meeting.wav --output meeting.json --meeting-id m042 \\
        --no-meta-extract --title "新機能企画" --goal "リリース範囲を決める"

    # config.yaml のパスを上書き / 話者数を固定
    python -m app.asr.cli --input audio.wav --output out.json \\
        --config /path/to/config.yaml --num-speakers 3
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

from app.asr.base import Transcriber
from app.asr.media import VIDEO_EXTENSIONS, MediaError, extract_audio_from_video, normalize_to_wav
from app.asr.meeting_builder import MetaExtractor, OpenAIMetaExtractor, build_meeting_input
from app.asr.pipeline import AudioPipeline
from app.asr.pyannote_diarizer import PyannoteConfig, PyannoteDiarizer
from app.asr.segmenter import merge_same_speaker_segments
from app.asr.volume_analyzer import LibrosaVolumeAnalyzer, VolumeThresholds
from app.asr.whisperx_transcriber import WhisperXConfig, WhisperXTranscriber
from app.schemas.models import MeetingInput
from app.scoring.weights import AppConfig, load_config

logger = logging.getLogger(__name__)


def _build_components(
    cfg: AppConfig,
    audio_cfg: dict[str, Any],
    use_meta_extractor: bool,
    use_volume_analyzer: bool,
) -> tuple[Transcriber, PyannoteDiarizer, LibrosaVolumeAnalyzer | None, MetaExtractor | None]:
    """config から ASR / Diar / Volume / MetaExtractor を組み立てる。"""
    asr_cfg = audio_cfg.get("asr", {})
    transcriber = WhisperXTranscriber(
        WhisperXConfig(
            model_name=asr_cfg.get("model_name", "large-v3"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            language=asr_cfg.get("language", "ja"),
            batch_size=int(asr_cfg.get("batch_size", 16)),
            align_model=asr_cfg.get("align_model"),
        )
    )

    diar_cfg = audio_cfg.get("diarization", {})
    diarizer = PyannoteDiarizer(
        PyannoteConfig(
            model_name=diar_cfg.get("model_name", "pyannote/speaker-diarization-3.1"),
            device=diar_cfg.get("device", "cuda"),
            hf_token_env=diar_cfg.get("hf_token_env", "HUGGINGFACE_HUB_TOKEN"),
            default_num_speakers=diar_cfg.get("default_num_speakers"),
            detect_overlap=bool(diar_cfg.get("detect_overlap", True)),
        )
    )

    volume_analyzer: LibrosaVolumeAnalyzer | None = None
    vol_cfg = audio_cfg.get("volume", {})
    if use_volume_analyzer and vol_cfg.get("enabled", True):
        volume_analyzer = LibrosaVolumeAnalyzer(
            VolumeThresholds(
                silent_below=float(vol_cfg.get("silent_below", 0.005)),
                low_below=float(vol_cfg.get("low_below", 0.02)),
                high_above=float(vol_cfg.get("high_above", 0.10)),
            )
        )

    meta_extractor: MetaExtractor | None = None
    if use_meta_extractor:
        meta_extractor = OpenAIMetaExtractor(
            model=cfg.llm_model,
            endpoint=cfg.llm_endpoint,
            api_key=cfg.llm_api_key,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
        )

    return transcriber, diarizer, volume_analyzer, meta_extractor


def transcribe_to_meeting_input(
    audio_path: Path,
    meeting_id: str,
    *,
    cfg: AppConfig,
    audio_cfg: dict[str, Any],
    num_speakers: int | None = None,
    use_meta_extractor: bool = True,
    use_volume_analyzer: bool = True,
    default_title: str = "",
    default_goal: str = "",
) -> MeetingInput:
    """音声 1 本を MeetingInput に変換する高レベル関数。

    `/upload_audio` エンドポイントと CLI の両方から呼ばれる本体。
    """
    transcriber, diarizer, volume_analyzer, meta_extractor = _build_components(
        cfg, audio_cfg, use_meta_extractor, use_volume_analyzer
    )
    pipeline = AudioPipeline(
        transcriber=transcriber,
        diarizer=diarizer,
        volume_analyzer=volume_analyzer,
        overlap_iou_threshold=float(
            audio_cfg.get("pipeline", {}).get("overlap_iou_threshold", 0.3)
        ),
    )
    raw_utterances = pipeline.run(audio_path, num_speakers=num_speakers)
    merged = merge_same_speaker_segments(raw_utterances, max_silence_sec=3.0)
    return build_meeting_input(
        merged,
        meeting_id=meeting_id,
        meta_extractor=meta_extractor,
        default_title=default_title,
        default_goal=default_goal,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.asr.cli",
        description="音声ファイルを MeetingInput JSON に変換する (Issue #11)",
    )
    parser.add_argument("--input", required=True, help="入力音声 (wav/mp3/m4a)")
    parser.add_argument("--output", required=True, help="出力 JSON パス")
    parser.add_argument("--meeting-id", required=True, help="会議 ID (例: m042)")
    parser.add_argument("--config", help="config.yaml のパス (既定: backend/config.yaml)")
    parser.add_argument("--num-speakers", type=int, help="話者数を強制指定 (None=自動)")
    parser.add_argument("--title", default="", help="default_title (LLM 抽出失敗時に使う)")
    parser.add_argument("--goal", default="", help="default_goal (LLM 抽出失敗時に使う)")
    parser.add_argument("--no-meta-extract", action="store_true", help="LLM メタ抽出をスキップ")
    parser.add_argument("--no-volume", action="store_true", help="音量分析 (librosa) をスキップ")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログを出力")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config) if args.config else load_config()
    audio_cfg = _load_audio_section(args.config)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("入力ファイルが見つかりません: %s", input_path)
        return 1

    # Issue #68: 動画拡張子ならローカル ffmpeg で音声抽出して既存パイプラインに渡す。
    # 抽出直後の webm を直接 pyannote に渡すと telemetry が duration=None で落ちる
    # ため、backend API と同じく normalize_to_wav で 16kHz mono wav に揃えてから ASR/Diar へ。
    audio_path = input_path
    extracted_tmp: Path | None = None
    normalized_tmp: Path | None = None
    tmp_dir: Path | None = None
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        logger.info("Video input detected. Extracting audio via ffmpeg ...")
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="asr_cli_extract_"))
            extracted_tmp = tmp_dir / f"{input_path.stem}.webm"
            extract_audio_from_video(input_path, extracted_tmp)
            logger.info(
                "Extracted audio: %s (%.1f MB)",
                extracted_tmp,
                extracted_tmp.stat().st_size / (1024 * 1024),
            )
            normalized_tmp = tmp_dir / f"{input_path.stem}.wav"
            normalize_to_wav(extracted_tmp, normalized_tmp)
            audio_path = normalized_tmp
        except MediaError as e:
            logger.error("動画→音声抽出/正規化に失敗: %s", e)
            return 2

    logger.info("Transcribing %s ...", audio_path)
    try:
        mi = transcribe_to_meeting_input(
            audio_path,
            meeting_id=args.meeting_id,
            cfg=cfg,
            audio_cfg=audio_cfg,
            num_speakers=args.num_speakers,
            use_meta_extractor=not args.no_meta_extract,
            use_volume_analyzer=not args.no_volume,
            default_title=args.title,
            default_goal=args.goal,
        )
    finally:
        # 動画抽出時の tmp を片付ける
        for p in (extracted_tmp, normalized_tmp):
            if p and p.exists():
                try:
                    p.unlink()
                except OSError:
                    logger.warning("tmp の削除に失敗 (放置可): %s", p)
        if tmp_dir and tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                logger.warning("tmp dir の削除に失敗 (放置可): %s", tmp_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        mi.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s (utterances=%d, title=%r)",
        out_path,
        len(mi.utterances),
        mi.title,
    )
    return 0


def _load_audio_section(config_path: str | None) -> dict[str, Any]:
    """config.yaml の `audio:` セクションだけを生 dict で読む。

    `load_config` は LLM/重み等を AppConfig に詰めるが、`audio:` は本実装で
    扱う領域なので、ここで別に dict として取り出す。
    """
    import yaml

    path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent.parent.parent / "config.yaml"
    )
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return raw.get("audio") or {}


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "transcribe_to_meeting_input",
]

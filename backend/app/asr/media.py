"""音声・動画のメディア処理ヘルパ (Issue #68)。

役割:
- `normalize_to_wav`: 任意の音声フォーマット (webm/opus, m4a/aac, mp3, ogg, wav 等) を
  ASR 用の 16kHz mono wav に正規化する。
- `extract_audio_from_video`: 動画ファイル (mp4 / mov / mkv 等) から音声トラックを
  軽量フォーマット (webm/opus or wav) として抽出する。

設計指針 (Issue #68 受け入れ条件より):
- backend は **音声の正規化担当** であり、動画ファイルを直接受け取らない経路を保つ
- 動画 → 音声抽出は **frontend で実行する** のが主経路。本モジュールの
  `extract_audio_from_video` は CLI (ローカル処理) からのみ呼ばれる想定
- ffmpeg が未導入の環境では `MediaError` で明確に失敗させる
- 動画に音声トラックが無い場合も `MediaError` で原因を出す
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# 音声入力として許容する拡張子。`audio_routes.py` と CLI で共有する。
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
# webm は音声コンテナとしても動画コンテナとしても使われる両用拡張子。
# Issue #68 では「backend に来た .webm は音声扱い」とする (frontend 抽出後の典型形式)。
AUDIO_EXTENSIONS_INCLUDING_WEBM = AUDIO_EXTENSIONS | {".webm"}

# 動画入力として認識する拡張子。**CLI 専用** (backend API はこれらを 415 で弾く)。
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


class MediaError(RuntimeError):
    """ffmpeg 関連のエラーをまとめて表す例外。"""


def _ensure_ffmpeg() -> str:
    """ffmpeg バイナリのパスを返す。未導入なら MediaError。"""
    path = shutil.which("ffmpeg")
    if not path:
        msg = (
            "ffmpeg が見つかりません。OS のパッケージマネージャでインストールしてください: \n"
            "  - macOS: brew install ffmpeg\n"
            "  - Ubuntu/Debian: sudo apt install ffmpeg\n"
        )
        raise MediaError(msg)
    return path


def _run_ffmpeg(args: list[str]) -> None:
    """ffmpeg を呼び出し、失敗時は stderr 末尾を要約して MediaError に包む。"""
    cmd = [_ensure_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args]
    logger.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as e:  # pragma: no cover - 通常起きない
        msg = f"ffmpeg の起動に失敗しました: {e}"
        raise MediaError(msg) from e
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
        msg = "ffmpeg 失敗 (exit={code}):\n  {tail}".format(
            code=proc.returncode,
            tail="\n  ".join(stderr_tail) or "(no stderr)",
        )
        raise MediaError(msg)


def normalize_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """任意の音声フォーマットを 16kHz mono wav に正規化する。

    Issue #68: 動画拡張子が渡された場合は受け付けず `MediaError` を投げる。
    動画→音声抽出は `extract_audio_from_video` で別途行うこと。

    Parameters
    ----------
    input_path : Path
        入力音声ファイル (webm/opus, m4a/aac, mp3, ogg, wav 等)。
    output_path : Path
        出力先。既存ファイルは上書き。
    sample_rate : int
        既定 16000 (Whisper / WhisperX の標準入力)。
    channels : int
        既定 1 (mono)。

    Returns
    -------
    Path : output_path (引数のまま、便利のため返す)。
    """
    if not input_path.exists():
        msg = f"入力ファイルが存在しません: {input_path}"
        raise MediaError(msg)
    suffix = input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        msg = (
            f"動画拡張子 ({suffix}) は normalize_to_wav では受け付けません。"
            "Issue #68 の方針に従い、frontend で音声抽出してから渡してください。"
            "CLI からなら extract_audio_from_video() を先に呼ぶこと。"
        )
        raise MediaError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vn",  # 動画ストリームを除外
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        msg = f"正規化後の wav が空または存在しません: {output_path}"
        raise MediaError(msg)
    return output_path


def extract_audio_from_video(
    input_path: Path,
    output_path: Path,
    *,
    codec: str = "libopus",
    bitrate: str = "32k",
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """動画ファイルから軽量音声を抽出する (CLI 専用)。

    Issue #68 受け入れ条件に従い、backend API はこの関数を呼ばない
    (動画は 415 で拒否される)。本関数は CLI からのローカル処理用。

    既定で webm/opus にエンコードする (frontend での想定形式に合わせる)。
    `output_path` の拡張子から ffmpeg がコーデックを推定するが、明示的に
    `-c:a` で `codec` を指定する。

    Parameters
    ----------
    input_path : Path
        入力動画 (mp4 / mov / mkv 等)。
    output_path : Path
        出力音声ファイル。.webm 推奨、.m4a も可。
    codec : str
        既定 "libopus"。.m4a なら "aac" を指定。
    bitrate : str
        既定 "32k" (Whisper にとっては十分、サイズ最小化)。
    sample_rate : int
        既定 16000。
    channels : int
        既定 1 (mono)。

    Returns
    -------
    Path : output_path。

    Raises
    ------
    MediaError : 動画に音声トラックが無い場合 / ffmpeg 失敗時。
    """
    if not input_path.exists():
        msg = f"入力ファイルが存在しません: {input_path}"
        raise MediaError(msg)
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS and suffix != ".webm":
        msg = f"動画拡張子ではありません ({suffix})。許容: {sorted(VIDEO_EXTENSIONS)}"
        raise MediaError(msg)

    if not has_audio_stream(input_path):
        msg = (
            f"動画 {input_path} に音声トラックが見つかりません。"
            "録画時に音声が記録されていない可能性があります。"
        )
        raise MediaError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vn",
            "-c:a",
            codec,
            "-b:a",
            bitrate,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        msg = f"抽出後の音声が空または存在しません: {output_path}"
        raise MediaError(msg)
    return output_path


def has_audio_stream(input_path: Path) -> bool:
    """`ffprobe` で動画/音声ファイルに音声ストリームがあるか確認する。

    `ffprobe` は ffmpeg と同梱されているのが普通。`-v error -select_streams a
    -show_entries stream=codec_type -of csv=p=0` で 'audio' が出力されれば真。
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # ffprobe が無くてもファイルが音声ファイルなら True 扱い (フォールバック)
        return input_path.suffix.lower() in (AUDIO_EXTENSIONS_INCLUDING_WEBM)
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(input_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:  # pragma: no cover
        return False
    return "audio" in (proc.stdout or "")


__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_EXTENSIONS_INCLUDING_WEBM",
    "VIDEO_EXTENSIONS",
    "MediaError",
    "extract_audio_from_video",
    "has_audio_stream",
    "normalize_to_wav",
]

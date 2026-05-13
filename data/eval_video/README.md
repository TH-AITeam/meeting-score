# data/eval_video: 動画→音声抽出ベンチ評価データ

Issue #68 (クライアント側音声抽出) のアップロード効率を計測するための動画ディレクトリ。

**動画ファイル本体はリポジトリに含めません** (容量 + プライバシー)。README と `.gitkeep` のみ commit し、各自のローカル / SSH 先で動画を準備して `scripts/run_video_benchmark.py` を回します。

## ディレクトリ構造（各自で用意する）

```
data/eval_video/
├── README.md                       # このファイル（commit 済み）
├── .gitkeep                        # ディレクトリ保持用（commit 済み）
├── meeting_01/                     # ★ 最低 1 本
│   └── video.mp4                   # 任意のフォーマット (.mp4 / .mov / .mkv / .avi / .webm)
├── meeting_02/
│   └── video.mov
└── ...
```

`meeting_*/` 以下は `.gitignore` で除外されます。

## 評価の意図 (Issue #68)

ブラウザ内 (`ffmpeg.wasm` / libopus / mono / 16kHz / 32kbps) で抽出した音声と、
オリジナル動画を比較し、以下を確認する:

1. **アップロード量の削減**: 動画 N MB → 音声 M MB の圧縮率
2. **抽出時間**: ローカル ffmpeg で同パラメータ実行した時間
   (ブラウザ ffmpeg.wasm はおよそ 2〜4 倍遅いので、これは下限値の目安)
3. **エンドツーエンドの WhisperX 結果が劣化しないこと** (任意)

## ベンチマーク実行

```bash
# 動画を data/eval_video/meeting_01/video.mp4 に置いた後
cd backend
uv run python ../scripts/run_video_benchmark.py
```

結果は `reports/video_benchmarks/upload_efficiency.json` に出ます。
表で見たい場合は `--print-table` を渡してください。

## 推奨ソース

- 公開された会議録画 (例: 国会中継、自治体公式アーカイブ、TED など)
- 自分で録画した社内ダミー会議 (個人情報・社外秘なし)
- TTS + 仮想カメラで合成した会議動画

オンライン社内会議や顧客との通話を**そのまま**評価データに使うことは避けてください。

## 関連

- Issue #68: 動画入力のクライアント側音声抽出
- `frontend/src/utils/audioExtract.ts`: ffmpeg.wasm 実装
- `backend/app/asr/media.py`: サーバー側 ffmpeg ヘルパ (CLI / ベンチ用)
- `scripts/run_video_benchmark.py`: 本ディレクトリを走査するベンチスクリプト

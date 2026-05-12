# data/eval_audio: 音声処理モデル評価データ

Issue #19 の音声処理モデル選定で使う評価音声を置くディレクトリ。

**音声ファイルそのものはリポジトリに含めません** (プライバシー + 容量)。本ディレクトリには README と `.gitkeep` のみコミットし、各自の Mac / SSH 先で評価データを準備して `scripts/run_audio_benchmark.sh` を回します。

## ディレクトリ構造（各自で用意する）

```
data/eval_audio/
├── README.md                       # このファイル（commit 済み）
├── .gitkeep                        # ディレクトリ保持用（commit 済み）
├── meeting_01/                     # ★ 最低 3 本必要
│   ├── audio.wav                   # 録音または合成、16kHz mono を推奨、15 分以上
│   ├── reference.txt               # 正解文字起こし全文
│   ├── speakers.rttm               # 正解話者ラベル (pyannote 標準 RTTM 形式)
│   └── named_entities.txt          # 固有名詞 (1 行 1 語、任意)
├── meeting_02/
│   └── ...
└── meeting_03/
    └── ...
```

`*` 以下のファイルは `.gitignore` で除外されるので、誤って commit してしまう心配はありません（追加で `git add -f` をしない限り）。

## Issue #19 の要求条件

| 項目 | 条件 |
|---|---|
| 本数 | 最低 **3 本** |
| 話者数 | 2〜4 人 |
| 長さ | 各 **15 分以上** |
| 形式 | 会議形式（議題を持ったやり取り）|
| 取得元 | **自作録音 または 合成音声**。**本物の業務会議は使わない** |

## プライバシー指針 (必読)

1. **本物の業務会議録音は使用禁止**: 参加者全員から書面同意を取った場合のみ例外。理由: 採用モデル選定のために録音が誰の目に触れるか管理しきれない
2. **個人情報・社外秘を含めない**: 自作録音であっても、参加者の本名・取引先名・顧客名・未公開のプロダクト名等は避ける。固有名詞認識率の評価は **合成・偽名で代替**
3. **合成音声を推奨**: 会議シナリオを書いて TTS (例: VOICEVOX, Style-Bert-VITS2 等) で多話者音声を生成すると、安全性とラベルの正確性が両立する
4. **`*.wav` / `*.mp3` / `*.m4a` などはリポジトリ root の `.gitignore` で除外**: PR に音声を含めない。SSH 先と Mac の間は scp / rsync で個別に転送する

## 推奨: 評価音声の作り方 (どちらか)

### A) 自分で録音

1. 会議シナリオを書く (15 分 × 3 本)
2. 自分 + 知人 2〜3 人（同意取得済み）で読み上げ録音
3. Audacity 等で mono 16kHz の wav に変換
4. 録音時に **誰がいつ喋ったか** を別途メモして RTTM を手作成
5. 録音音声を聞き直して `reference.txt` を手作成

### B) 合成音声 (推奨)

1. 会議シナリオを書く（話者ラベル付き）。例:
   ```
   [00:00:00-00:00:08] SPEAKER_00: では始めましょう。今日の議題は新機能の優先順位です。
   [00:00:08-00:00:15] SPEAKER_01: 私からは UI 改修案を 3 つ持ってきました。...
   ```
2. TTS (VOICEVOX / Style-Bert-VITS2 / fish-speech 等) で話者ごとに別音声を生成
3. ffmpeg / sox で時刻合わせして 1 本の wav に結合
4. シナリオから `reference.txt` (全文) と `speakers.rttm` (時刻+話者) を自動生成できる

## reference.txt の形式

正解文字起こしの **全文** をプレーンテキストで書く。発言の区切りはあっても無くてもよい（CER は文字単位）。

```
では始めましょう。今日の議題は新機能の優先順位です。私からは UI 改修案を 3 つ持ってきました。...
```

## speakers.rttm の形式

[RTTM (Rich Transcription Time Marked)](https://github.com/nryant/dscore#rttm) 形式。pyannote が標準で読み書きできます。

```
SPEAKER meeting_01 1 0.000 8.000 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting_01 1 8.000 7.000 <NA> <NA> SPEAKER_01 <NA> <NA>
```

各列: `SPEAKER <file_id> <channel> <start_sec> <duration_sec> <NA> <NA> <speaker_label> <NA> <NA>`

合成音声の場合は、シナリオから自動生成スクリプトを書ける（数十行）。

## named_entities.txt の形式（任意）

固有名詞認識率を測るための **会議に登場する固有名詞** を 1 行 1 語で並べる。

```
プロジェクトX
山田太郎
APIエンドポイント
```

ファイルが無い場合、固有名詞認識率は計測されず `null` で記録される。

## ベンチマーク実行

評価音声を 3 本以上配置したら:

```bash
# SSH 先 (RTX 5090) で
source backend/.venv/bin/activate
uv pip install # ※ uv sync --extra audio で whisperx / pyannote / librosa 導入済みの前提
export HUGGINGFACE_HUB_TOKEN=hf_xxxxx
bash scripts/run_audio_benchmark.sh --all
```

結果は `reports/audio_benchmarks/{model_id}/{asr,diar}.json` に出ます。`docs/audio_model_selection_v1.md` の各表に手動転記してください。

## 関連

- Issue #19: 音声処理モデル選定
- `docs/audio_model_candidates.md`: 候補モデル
- `docs/audio_model_selection_v1.md`: スコアシート (TBD)
- `docs/adr/0002-audio-model.md`: 採用判断

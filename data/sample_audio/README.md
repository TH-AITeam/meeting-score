# data/sample_audio: CLI / `/upload_audio` の動作確認用サンプル音声

Issue #11 の音声入力パイプラインの動作確認用に置くサンプル音声ディレクトリ。`data/eval_audio/` (Issue #19 のベンチ用) とは別物です。

**音声ファイルはリポジトリに含めません** (プライバシー + 容量)。本ディレクトリには README と `.gitkeep` のみ commit し、各自で 1〜2 本の短い音声 (1〜5 分程度) を配置して、CLI / FastAPI エンドポイントが動くことを確認します。

## ディレクトリ構造（各自で用意する）

```
data/sample_audio/
├── README.md             # このファイル
├── .gitkeep              # ディレクトリ保持用
└── sample_01.wav         # 短い動作確認音声 (1〜5 分、自作録音 or 合成、任意名)
```

`.wav` `.mp3` `.m4a` `.flac` は `.gitignore` で除外済み。誤ってコミットされません。

## 用途

| ファイル | 用途 |
|---|---|
| `data/sample_audio/` | CLI / API の動作確認 (短い音声、本数は問わない) |
| `data/eval_audio/` | Issue #19 のベンチマーク (15 分以上 × 3 本、正解付き) |

両方とも自作録音または合成音声で、業務会議の生録音は禁止 (プライバシー)。

## 動作確認手順

```bash
# 1. サンプル音声を 1 本配置
cp ~/Downloads/test.wav data/sample_audio/sample_01.wav

# 2. CLI で MeetingInput JSON に変換 (LLM メタ抽出は別途 vLLM が必要)
cd backend && python -m app.asr.cli \
    --input ../data/sample_audio/sample_01.wav \
    --output /tmp/test_meeting.json \
    --meeting-id m_test \
    --no-meta-extract \
    --title "テスト会議" --goal "動作確認"

# 3. 結果を見る
cat /tmp/test_meeting.json | head -40
```

メタ抽出も含めて完全な MeetingInput を作りたい場合は、SSH 先で vLLM を立てておくか、`config.yaml` の `llm.backend=openai` で OpenAI API を使うこと。

## FastAPI 経由での動作確認

```bash
# サーバ起動
cd backend && uv run uvicorn app.api.main:app --reload --port 8000

# 別ターミナルから
curl -X POST http://localhost:8000/api/upload_audio \
    -F "file=@data/sample_audio/sample_01.wav" \
    -F "meeting_id=m_test" \
    -F "no_meta_extract=true" \
    -F "title=テスト" \
    -F "goal=動作確認"
```

## 関連

- Issue #11: 音声入力パイプライン
- Issue #19 / `data/eval_audio/`: 精度評価ベンチマーク
- `backend/app/asr/`: 本実装
- `backend/prompts/meta_extraction.txt`: メタ抽出プロンプト

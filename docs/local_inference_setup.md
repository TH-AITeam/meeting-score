# ローカル推論セットアップ手順 (Issue #12)

OpenAI 依存を捨て、Lyon の A100 上で判断モデルを動かすためのセットアップ手順。

## 全体像

```
[Streamlit / FastAPI]
        │
        │ OpenAI 互換 SDK
        │ base_url=http://lyon:8001/v1
        ▼
[vLLM サーバ on Lyon A100]
        │
        │ HF Hub / ローカル重み
        ▼
[判断モデル (Qwen2.5-7B-Instruct 等)]
```

## 1. サーバ側 (Lyon)

### 依存インストール

```bash
# uv 推奨。conda でも可
uv pip install vllm
# あるいは
pip install "vllm>=0.6"
```

### 起動

```bash
# 既定モデルでフォアグラウンド起動（動作確認用）
bash scripts/serve_local_llm.sh

# 別モデルで
MODEL="meta-llama/Llama-3.1-8B-Instruct" bash scripts/serve_local_llm.sh
```

### 起動確認

```bash
curl -s http://localhost:8001/v1/models | jq .
```

`data[0].id` に MODEL 名が出ていれば OK。

### systemd で常駐化（推奨）

`/etc/systemd/system/meeting-score-llm.service`:

```ini
[Unit]
Description=meeting-score local LLM (vLLM)
After=network.target

[Service]
User=tomoki
WorkingDirectory=/home/tomoki/meeting-score
Environment=MODEL=Qwen/Qwen2.5-7B-Instruct
Environment=PORT=8001
Environment=GPU_MEM_UTIL=0.85
ExecStart=/bin/bash scripts/serve_local_llm.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now meeting-score-llm
sudo systemctl status meeting-score-llm
```

### Docker で動かす場合

既存の Docker 環境がある場合は、以下のような compose を `infra/compose.local-llm.yml` に
追記する想定（本 PR ではテンプレートのみ）:

```yaml
services:
  local-llm:
    image: vllm/vllm-openai:latest
    command:
      - --model=Qwen/Qwen2.5-7B-Instruct
      - --guided-decoding-backend=xgrammar
      - --gpu-memory-utilization=0.85
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped
```

## 2. クライアント側 (本リポジトリ)

`config.yaml` を編集:

```yaml
llm:
  backend: "local"
  model: "Qwen/Qwen2.5-7B-Instruct"     # サーバが配信しているモデル名と一致
  endpoint: "http://lyon:8001/v1"        # サーバの公開アドレス
  api_key: null                           # vLLM は不要
  max_tokens: 1024
  max_retries: 3
  timeout: 30
```

ローカル開発時:

```bash
# 1. Lyon と SSH ポートフォワード
ssh -L 8001:localhost:8001 lyon

# 2. config.yaml で endpoint: "http://localhost:8001/v1"
# 3. アプリ起動
uv run python -m uvicorn app.api.main:app --reload
```

## 3. 動作確認

```bash
# サンプル分析を投げる
curl -s -X POST http://localhost:8000/api/analyze/sample/sample_meeting_01.json | jq '.evaluated_utterances[0]'
```

`speech_type` / `scores` / `penalties` が正しい列挙値・範囲で返ってくれば OK。

## 4. レイテンシ / スループット計測

```bash
# 1発言推論を100回連続実行して p50 / p95 を見る
# (TODO: scripts/bench_local_llm.py を別 PR で追加予定)
```

Issue #12 の完了条件「1発言あたり <2s」を満たすかは Issue #17 のモデル確定後に
本ドキュメントの末尾に追記する。

## トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| `400 Bad Request - guided_json not supported` | `--guided-decoding-backend xgrammar` を付けて起動 |
| `OOM` | `--gpu-memory-utilization` を 0.7 程度に下げる、または小さいモデルに切替 |
| OpenAI SDK が `connection refused` | endpoint の host/port を再確認、firewall / SSH トンネル |
| `model not found` | サーバ側 `--served-model-name` と config.yaml の model を一致させる |

## 関連

- `scripts/serve_local_llm.sh` : 起動スクリプト
- `app/evaluators/local_evaluator.py` : クライアント実装
- `docs/inference_server_selection.md` : vLLM 採用理由とベンチ結果
- Issue #17 : 採用モデル選定
- Issue #16 : OpenAI 依存の完全撤去

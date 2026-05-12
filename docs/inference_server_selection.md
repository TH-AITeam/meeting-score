# 推論サーバ選定 (Issue #12)

判断モデルが Issue #18 で決まったあと、それを動かす推論サーバを比較するためのドキュメント。
（issue ファイル `17_judgment_model_selection.md` の中身が GitHub Issue #18 に対応する経緯のため、
古い記述で「Issue #17」と書かれている箇所は **判断モデル選定 = Issue #18** を指す。）

## 比較軸

| 軸 | 説明 |
| --- | --- |
| **採用モデルへの対応状況** | Issue #18 で選定したモデル（Qwen3.6 / Qwen3 / Llama3.3 / Swallow / Phi-4）が動くか |
| **guided decoding (JSON 強制)** | `response_format=json_schema` 相当が使えるか |
| **スループット / レイテンシ** | A100 1枚で 8B クラスを動かしたときの req/s と median |
| **運用容易さ** | Docker / systemd 起動、ヘルスチェック、モデル切替の手順 |

## 候補サーバ

### vLLM（既定候補）

- ✅ OpenAI 互換 `/v1/chat/completions` をネイティブ提供
- ✅ `--guided-decoding-backend xgrammar` で JSON Schema 強制
- ✅ continuous batching でスループット重視
- ✅ コミュニティ規模・採用実績が最大
- ⚠️ 起動は重い（モデルロードに数十秒）

### sglang

- ✅ OpenAI 互換、JSON 強制（regex / xgrammar）対応
- ✅ Chunked prefill、RadixAttention でレイテンシが速い場合あり
- ⚠️ コミュニティ規模は vLLM より小さい
- ⚠️ モデル対応の振れ幅がやや大きい

### TGI (Hugging Face)

- ✅ OpenAI 互換あり（ただし完全互換ではない）
- ✅ JSON 強制（Outlines / guided generation）対応
- ⚠️ 設定が他より複雑、Rust ベースのため起動の癖がある

## 暫定判断

**vLLM を第一候補** とする。理由:

1. OpenAI SDK との互換性が最も整っている（本 PR のクライアントはそのまま動く）
2. `--guided-decoding-backend xgrammar` で JSON Schema が確実に効く
3. Lyon の Docker / systemd 運用と相性が良い

ベンチマークは Issue #18 で採用モデル確定後、`scripts/run_model_benchmark.sh` 経由で
vLLM / sglang を起動して 1 発言推論を 100 回回し、p50 / p95 を計測する。
結果は本ドキュメントの「ベンチマーク結果」セクションに追記する。

## ベンチマーク結果

> Issue #18 のモデル確定後に追記。

| サーバ | モデル | p50 (ms) | p95 (ms) | req/s | JSON 強制 |
| --- | --- | --- | --- | --- | --- |
| vLLM | TBD | - | - | - | - |
| sglang | TBD | - | - | - | - |

## 関連

- Issue #11 → 本実装 (`app/evaluators/local_evaluator.py`)
- Issue #18 → 採用モデル名の決定（`docs/model_selection_v1.md`, `docs/adr/0001-judgment-model.md`）
- Issue #17 → OpenAI 依存の完全撤去

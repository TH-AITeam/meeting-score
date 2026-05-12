# #11 [feature] ローカル判断モデル推論基盤(vLLM サーバ)

**Labels**: `feature`, `infra`, `breaking-change`, `P0`
**Milestone**: v0.5

## 前提

**先に #17 で判断モデルを選定すること**。 ここでは「採用モデル」と抽象的に書く。

## 概要

OpenAI API 依存を捨て、判断モデル(#17 で選定)を Lyon の A100 上に立てる。**評価器の抽象化**と **推論サーバ運用**の2本立て。

## アーキテクチャ

```
app/evaluators/
  base.py              # Evaluator(ABC).evaluate(ctx) → EvaluationResult
  llm_evaluator.py     # 既存 OpenAI 実装 (deprecated)
  local_evaluator.py   # 新規: OpenAI 互換エンドポイント経由でローカルモデルを叩く
  factory.py           # config.llm.backend で切り替え
```

- vLLM や sglang など、**OpenAI 互換 API を提供する推論サーバ**を使えば既存コードの構造が大体使い回せる
- 軸スコア範囲の強制には guided decoding 系の機能を使う(`guided_json` / `xgrammar` 等、採用サーバの仕様に従う)

## 推論サーバの選定

判断モデルが決まったあとに、それを動かす推論サーバ(vLLM / sglang / TGI 等)も短く比較する。比較軸:

- 採用モデルへの対応状況
- guided decoding(JSON 強制)対応
- スループット / レイテンシ
- 運用の容易さ(再起動、ヘルスチェック、モデル切替)

結論は `docs/inference_server_selection.md` に残す。

## やること

- [ ] `Evaluator` 抽象クラスを `app/evaluators/base.py` に定義
- [ ] `LocalEvaluator` 実装(OpenAI 互換 API なので `openai` SDK で `base_url` 指定)
- [ ] `config.yaml` に `llm.backend`, `llm.endpoint`, `llm.model` を追加
- [ ] `factory.create_evaluator(config)` で切り替え
- [ ] JSON 強制(guided decoding)の動作確認
- [ ] 起動スクリプト `scripts/serve_local_llm.sh` を追加
- [ ] Lyon の Docker 環境(既存)に推論サーバコンテナを足す or systemd で常駐化
- [ ] レイテンシ・スループット計測を README に記載

## 完了条件

- `config.yaml` で `backend: local` に切り替えるだけで全テストが通る
- サンプル会議3本で OpenAI 実装と ローカル実装の Spearman ≥ 0.7 (#4 のメトリクスで)
- 1発言あたりの推論時間が <2s

## 補足

ここまでで「OpenAI 課金ゼロで動く」状態になる。次は #12 で学習データを作って自前学習に進む。

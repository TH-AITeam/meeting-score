# ADR 0001: 判断 LLM の採用モデル

- **Status**: Proposed（実測後に Accepted に更新）
- **Date**: 2026-05-12
- **Deciders**: TomokiAkiyama06
- **Related**: Issue #18, Issue #5, Issue #12, Issue #11/#13/#14/#16

## Context

会議貢献度スコアリング (#5 eval ハーネス、#13/#14 SFT/DPO、#11 ローカル推論) で使う **判断 LLM** を 1 つに決める必要がある。下流の SFT/DPO はモデルが決まらないと学習設定が書けない。一方で本プロジェクトの実行環境は **A100 80GB ではなく RTX 5090 32GB**（SSH 先）であり、issue 仕様の前提とは異なる。

## Decision

第一採用: **`Qwen/Qwen3.6-27B-Instruct-AWQ`** (AWQ INT4 量子化、vLLM 配信)
控え (fallback): **`Qwen/Qwen3-14B-Instruct`** (bf16)

> **Status は Proposed**。SSH 先 (RTX 5090) で `scripts/run_model_benchmark.sh --all`
> を回し、`docs/model_selection_v1.md` のスコアシートが埋まった時点で
> Accepted に切り替える（または再判定）。

## Options Considered

`docs/model_candidates.md` で詳述した 5 候補:

1. **Qwen3.6-27B-Instruct (AWQ)** ← 第一候補
2. **Qwen3-14B-Instruct (bf16)** ← 控え
3. **Llama-3.3-8B-Instruct (bf16)** — 対照群（英語ベース 8B）
4. **Swallow-Llama-3.1-8B-Instruct (bf16)** — 対照群（日本語特化 8B）
5. **Phi-4-14B (bf16)** — 対照群（同サイズ別系統）

落選: Gemma 2/3 (ライセンス制約)、Plamo-100B (サイズ超過)、Mistral Small 3 (vLLM AWQ 初期不安定)、DeepSeek-V2-Lite MoE (PEFT 実装の成熟度不足)。

## Rationale

判定軸（重み付き）と Qwen3.6-27B を第一候補にする理由:

1. **日本語精度（重み 0.40）**: 公開ベンチ（JMT-Bench, Nejumi LLM Leaderboard, 2026 初頭時点）で Qwen3 系が 30B 以下クラス最上位。Llama-3.3 系より日本語タスクで有意な差
2. **JSON 出力安定性（0.10）**: Qwen3 系は tool call / structured output の訓練データが厚く、xgrammar guided JSON との相性が良い実績
3. **学習しやすさ**: Apache 2.0 で SFT/DPO/蒸留に制約なし。HF transformers + peft + bitsandbytes の QLoRA 例が豊富
4. **メモリ占有（0.10）**: AWQ INT4 で実 GPU 使用量 ~14GB、RTX 5090 32GB に対して KV cache 16K トークン分の余裕がある
5. **SFT 後の伸びしろ**: 同サイズの 8B Llama 系より、27B からの QLoRA の方が会議ドメインの差分学習で勝ちやすい（issue 仕様の注意点に合致）

控えに Qwen3-14B を置く理由:

- 27B が SFT で重すぎる（学習 1 step 数十秒）場合の draft / controller モデル
- 推論速度を稼ぎたい運用 (リアルタイム) に切り替えやすい
- 同系統なのでプロンプトテンプレと tokenizer を 27B と共有可能 → 切替コストが低い

## Consequences

採用後（Accepted）になった時点で:

- `backend/config.yaml.example` の `llm.model` を `qwen3.6-27b-awq` に固定
- Issue #11 (音声 → 判断 LLM への接続) は本モデルを前提に実装開始
- Issue #13 (SFT データセット) は Qwen3.6 の chat template に合わせる
- Issue #14 (SFT) は QLoRA on Qwen3.6-27B-AWQ の学習スクリプトを書く
- Issue #16 (OpenAI 撤去) は本モデルが production 安定後に着手
- `docs/model_history.md` に v1 採用エントリ（モデル名・採用日・代替候補）を追加

将来の見直しトリガ:

- 半年〜1年に 1 回、または上位の OSS モデル（30B クラスで日本語強い）が出た時
- 採用モデルの推論速度が会議リアルタイム要件を満たさなくなった時
- ライセンス変更で再配布／蒸留が制約された時

## Open Questions

- [ ] vLLM 0.6+ で sm_120 (Blackwell) の AWQ Marlin カーネルが安定動作するか
  →初回 `run_model_benchmark.sh` 実行時に確認
- [ ] Qwen3.6-27B-AWQ の HF 配布が AWQ-quantized 公式で出ているか、サードパーティのみか
  → 出ていなければ `casperhansen/qwen3.6-27b-awq` 等の有志公開版を使う or 自前で AWQ
- [ ] Swallow が AWQ 量子化未対応の場合、bf16 16GB のままで KV cache をどこまで取れるか

## References

- `docs/model_candidates.md`: 候補絞り込みの詳細
- `docs/model_selection_v1.md`: 総合スコアシート（実測転記先）
- `docs/inference_server_selection.md`: vLLM 採用根拠 (#12)
- Issue #18: 本 Issue

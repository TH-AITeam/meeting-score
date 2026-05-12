# ADR 0001: 判断 LLM の採用モデル

- **Status**: Proposed（精度測定（Issue #6 アノテ完成後）で Accepted に更新予定）
- **Date**: 2026-05-12
- **Deciders**: TomokiAkiyama06
- **Related**: Issue #18, Issue #5, Issue #6, Issue #12, Issue #11/#13/#14/#17

## Context

会議貢献度スコアリングのコア LLM をローカル運用前提で 1 つに決める必要がある。下流の SFT/DPO (#13/#14)、OpenAI 撤去 (#17)、音声入力統合 (#11) はすべて採用モデルに依存する。実行環境は **A100 80GB ではなく RTX 5090 32GB**（SSH 先、Blackwell sm_120）であり、issue 仕様の前提とは異なる。

実機ベンチを `scripts/run_model_benchmark.sh --all` で 6 候補に対して実行し、レイテンシ / JSON 成功率 / 安定性 (greedy) を実測した。Spearman / pairwise accuracy は Issue #6 のゴールデンアノテ完成後に追加測定する。

## Decision

第一採用 (Proposed): **`unsloth/Qwen3.6-35B-A3B-NVFP4`**
控え (fallback): **`Qwen/Qwen2.5-32B-Instruct-AWQ`** (AWQ Marlin)

`docs/model_selection_v1.md` のスコアシート参照。**Status は Proposed**。精度系メトリクスが Issue #6 完成後に判明し、結果次第で:

- 精度差が小さければ → 控え (Qwen2.5-32B-AWQ) に切替（レイテンシ 2 倍高速のため運用優位）
- 精度差が大きければ → 第一採用 (Qwen3.6-35B-A3B NVFP4) を Accepted に確定

## Options Considered

`docs/model_candidates.md` で詳述した 6 候補（HF API で実在確認・実機ロード確認済み）:

1. **`unsloth/Qwen3.6-35B-A3B-NVFP4`** ← 第一候補（NVFP4 / MoE / Apache 2.0）
2. **`Qwen/Qwen2.5-32B-Instruct-AWQ`** ← 控え（AWQ Marlin / 最速）
3. `Qwen/Qwen3.6-27B` (BnB NF4) — dense 比較
4. `Qwen/Qwen3-14B` (BnB NF4)
5. `microsoft/phi-4` (BnB NF4)
6. `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3` (bf16)

落選: 本家 `Qwen/Qwen3.6-35B-A3B` の素モデル / GGUF / 標準 BnB は vLLM 0.20.2 が `qwen35moe` 未対応で動作不可。Qwen3.6-27B-Instruct(-AWQ) は HF 未公開。Llama-3.3-8B は実在せず。Gemma 2/3 はライセンス制約。

## Rationale

### Qwen3.6-35B-A3B (NVFP4) を第一候補にする理由

1. **MoE による知識量と推論コストのバランス**: 35B 総パラメータ / 3B アクティブ。SFT/DPO のターゲットとして 14B〜27B dense より知識量が豊富
2. **Blackwell ネイティブ FP4**: 5090 (sm_120) の FP4 ハードウェアを使い切る量子化形式。今後 vLLM の FP4 / MoE 最適化が進めばレイテンシも改善見込み
3. **Apache 2.0**: SFT/DPO/蒸留に制約なし。`unsloth` の量子化済みチェックポイントも Apache 2.0 で再配布可
4. **32GB に余裕で乗る**: NVFP4 で 23GB on disk、GPU 上もほぼ同等。KV cache fp8 で 4K コンテキスト分確保可能
5. **動作実証済み**: 本家 MoE は vLLM 未対応で全滅したが、`compressed-tensors` フォーマットの NVFP4 だけが起動・推論成功

### Qwen2.5-32B-Instruct-AWQ を控え候補にする理由

- **レイテンシ最速 (p50 3.0s)**: AWQ Marlin カーネル最適化が効く。35B-A3B NVFP4 (p50 7.4s) の半分以下
- **dense アーキで安定**: MoE のルーティングが学習で不安定化するリスクを避けたい場合の保険
- **Apache 2.0**: 第一候補と同じライセンス自由度
- **検証済み**: 公式 AWQ で初回ロードが速く、JSON 強制も安定

### 採用判定の重み配分

精度 40%、pairwise 20%、Top5 10%、JSON 成功率 10%、mean SD 10%、p95 5%、ライセンス 5%（`docs/model_selection_v1.md` 参照）。

## Consequences

採用後（Accepted）になった時点で:

- `backend/config.yaml.example` の `llm.model` を採用モデルに固定 (現状は `qwen3.6-35b-nvfp4` 仮設定)
- Issue #11 (音声 → 判断 LLM 接続) は本モデルを前提に実装開始
- Issue #13 (SFT データセット) は採用モデルの chat template に合わせる
- Issue #14 (SFT) は QLoRA on 採用モデルの学習スクリプトを書く
- Issue #17 (OpenAI 撤去) は本モデルが production 安定後に着手
- `docs/model_history.md` に v1 採用エントリ（モデル名・採用日・代替候補）を追加

将来の見直しトリガ:

- 半年〜1年に 1 回の棚卸し
- vLLM が `qwen35moe` アーキを公式対応 → 素の `Qwen/Qwen3.6-35B-A3B` を直接使えるか再評価
- Qwen3.6 系の公式 AWQ チェックポイントが公開 → NVFP4 から AWQ Marlin に置き換え検討
- 採用モデルの推論速度が会議リアルタイム要件を満たさなくなった時
- ライセンス変更で再配布／蒸留が制約された時

## Open Questions

- [ ] **Spearman / pairwise accuracy の実測**: Issue #6 アノテ完成後に `make eval DATASET=data/annotations/gold/v1` で実施
- [ ] **真の安定性測定**: 現状は greedy decode の確認だけ。`LocalEvaluator` に `temperature` 引数を追加して N=5 採点の本来の SD を測る（別 Issue）
- [ ] **vLLM の Qwen35MoE 対応**: 本家 `Qwen/Qwen3.6-35B-A3B` (素 / GGUF / BnB) が動くようになれば、unsloth NVFP4 から本家への乗り換え可能。リリースをウォッチ
- [ ] **AWQ Marlin の sm_120 安定性**: Qwen2.5-32B-AWQ では問題なく動作。長期運用で問題が出るか継続観察

## References

- `docs/model_candidates.md`: 候補絞り込みの詳細
- `docs/model_selection_v1.md`: 総合スコアシート（実測値反映済み）
- `docs/inference_server_selection.md`: vLLM 採用根拠 (#12)
- `reports/model_benchmarks/`: 各モデルの生 JSON 出力
- Issue #18: 本 Issue

# 判断モデル採用履歴

判断 LLM の世代管理。採用モデルは固定せず、半年〜1年で見直す前提で履歴を残す。
変更時は ADR (`docs/adr/`) を追加し、本ファイルの表に行を追加する。

## 採用履歴

| 世代 | 採用日 | 採用モデル | 量子化 | 控え | 判断軸の重点 | ADR | 廃止日 |
|---|---|---|---|---|---|---|---|
| v1 | 2026-05-12 (Proposed) | `unsloth/Qwen3.6-35B-A3B-NVFP4` | NVFP4 (compressed-tensors) | `Qwen/Qwen2.5-32B-Instruct-AWQ` | Qwen3.6 MoE の動作実証済み唯一の量子化版 + Blackwell ネイティブ FP4 + Apache 2.0。本家 `Qwen3.6-35B-A3B` が vLLM 対応されたら乗り換え予定 | [ADR 0001](adr/0001-judgment-model.md) | – |

> 採用日は ADR が Proposed のときの日付。**Accepted に切り替わったらここを上書きする**。

## 見直しトリガ

- **時期**: 半年〜1年ごとに棚卸し
- **イベント駆動**:
  - 上位の OSS モデル（30B 級で日本語強い）が出た
  - 採用モデルの推論速度が会議リアルタイム要件を満たさなくなった
  - ライセンス変更で再配布／蒸留が制約された
  - eval ハーネスのベースラインが半年以上頭打ち

## 更新手順

1. 新候補を `docs/model_candidates.md` に追加（必要なら全面差し替え）
2. `scripts/run_model_benchmark.sh --all` で実測
3. `docs/model_selection_v1.md` を `docs/model_selection_v{N}.md` にコピー → 更新
4. ADR を追加（`docs/adr/000N-judgment-model.md`）
5. 本ファイルの表に新世代を追加し、旧世代の「廃止日」を記入
6. `backend/config.yaml.example` の `llm.model` を更新
7. 必要なら SFT/DPO の学習を新モデルでやり直す（#13/#14 の再走）

## 関連

- `docs/adr/`: 個別の意思決定記録
- `docs/model_candidates.md`: 現世代の候補絞り込み詳細
- `docs/model_selection_v{N}.md`: 世代ごとの総合スコアシート
- Issue #18: 初回世代 v1 の選定

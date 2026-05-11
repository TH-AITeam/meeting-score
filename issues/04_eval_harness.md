# #4 [feature] eval ハーネス整備

**Labels**: `infra`, `eval`, `P0`
**Milestone**: v0.2

## 概要

AGENT.md §15 の検証基準を **自動で測れる** 状態にする。学習を回す前にこれが先。「人が見て納得」を Spearman / Top-K Jaccard / ペアワイズ精度に分解して計測する。

## ディレクトリ構成

```
evals/
  __init__.py
  metrics.py            # Spearman, Kendall tau, Top-K Jaccard, pairwise accuracy
  runner.py             # run_eval(model_id, dataset_path) → metrics dict
  stability.py          # 同一発言を N 回採点して分散を測る
  cli.py                # `python -m evals.cli --dataset data/annotations/gold/v1 ...`
data/
  annotations/
    gold/
      v1/
        meeting_01_tags.jsonl
        meeting_01_pairs.jsonl
        meeting_01_top_bottom.jsonl
```

## やること

- [ ] `evals/metrics.py` 実装
  - `spearman(human_ranks, system_ranks)`
  - `top_k_jaccard(human_top, system_top, k=5)`
  - `pairwise_accuracy(pairs, system_scores)`  ← 「Aの方が貢献した」「Bの方が貢献した」「同等」の3クラス
- [ ] `evals/stability.py`
  - 同一発言を `temperature=0.7` で N=5 回採点
  - 7軸の標準偏差・最大-最小を出す
  - **個別発言レベル** と **会議レベル** の両方で
- [ ] `evals/runner.py`
  - 入力: アノテ済みデータセットパス
  - 出力: metrics dict + per-meeting breakdown
- [ ] `evals/cli.py` で CLI 化
- [ ] `Makefile` に `make eval DATASET=v1 MODEL=...` を追加

## 完了条件

- Issue #5, #6 のアノテーションが揃った段階で、`make eval` 一発で全 metric が出る
- 結果が JSON で `evals/results/<timestamp>_<model>.json` に保存される
- README に「ベースラインスコア」セクションが追加される

## 補足

Kaggle CV 設計の延長。fold ごとの分散を出す感覚で、会議ごとの分散も必ず出すこと。

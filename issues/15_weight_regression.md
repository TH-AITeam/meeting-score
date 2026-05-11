# #15 [training] 軸重みを人手データから回帰で出す

**Labels">: `training`, `P2`
**Milestone**: v0.6

## 概要

`config.yaml` の軸重み(`issue_clarification × 1.3` 等)は現状カン。Issue #5 の人手 Top-K / ペアワイズデータから **線形回帰** or **ロジスティック回帰** で重みを推定する。

## アプローチ

### A) ペアワイズロジスティック回帰

人手ペア `(A, B, winner)` に対し、軸スコアの差分ベクトル `x = score(A) - score(B)` を作り、`P(A wins) = sigmoid(w · x)` を最尤推定。

得られた `w` がそのまま `ScoringWeights` になる。

### B) Top-K 線形回帰

Top-K 発言の総合スコアが高くなる方向に、重み制約付き最小二乗。

A) の方が筋がいい（順位情報を直接使う）ので A) を主とする。

## やること

- [ ] `training/regress_weights.py`
  - 入力: `data/annotations/gold/v1/pairs.jsonl` + 全発言の軸スコア
  - sklearn `LogisticRegression` で `w` を推定
  - 制約: 加点軸の重みは ≥ 0、減点軸は ≥ 0 (penalty値が負なので符号で吸収)
  - 出力: `config/weights_regressed.yaml`
- [ ] eval で **固定重み vs 回帰重み** の Spearman 比較
- [ ] 結果が良ければ `config.yaml` のデフォルトに採用

## 完了条件

- 回帰重みで Spearman, ペアワイズ accuracy のどちらかが固定重みより +0.02 以上向上
- 結果が `docs/weight_regression_report.md` にまとめられる

## 注意

- アノテ数が少ない (< 100 ペア) と過学習する。Ridge 正則化を入れる
- 軸間の相関が高い軸は重みが不安定になる(VIF 確認)。前回提案した「軸の独立性」が効くポイント
- 結果次第では「論点整理 × 1.3 → 1.7」みたいに大きく動く可能性あり。一度ユーザーテストで確認

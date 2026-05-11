# #14 [training] DPO 学習スクリプト（ペアワイズ選好で順位の納得感を上げる）

**Labels**: `training`, `P1`
**Milestone**: v0.6

## 概要

SFT モデルに対して DPO (Direct Preference Optimization) を回す。Issue #5 の人手ペアワイズ + Issue #7 の合成ペアを使い、**順位の納得感**を直接学習する。

AGENT.md §15 「人間が見て『まあ納得』できるか」を、絶対スコア精度ではなく**ペアワイズで**最適化するのが筋。

## データ形式

```json
{
  "prompt": "<同じ評価プロンプト>",
  "chosen": "<better な発言評価JSON>",
  "rejected": "<worse な発言評価JSON>",
  "meta": {"source": "human|synthetic", "pattern": "verbose"}
}
```

ペアワイズデータを「対象発言が違う2件への評価結果」に変換するのがコツ。`chosen` は高評価が出るべき発言の JSON、`rejected` は低評価が出るべき発言の JSON。

## やること

- [ ] `scripts/build_dpo_dataset.py`
  - 人手ペアワイズ(#5) + 合成ペアワイズ(#7) → DPO 形式に変換
  - 各ペアに対し、SFT モデルで両方の評価 JSON を生成
  - `winner` 側を chosen、 loser 側を rejected
- [ ] `training/configs/dpo_v1.yaml`
  - β=0.1, lr=5e-7, epochs=1
- [ ] `training/dpo_train.py`(`trl.DPOTrainer`)
- [ ] SFT モデルから初期化、DPO 学習
- [ ] eval で SFT 単独 vs SFT+DPO を比較

## 完了条件

- ペアワイズ accuracy が SFT-only より +3pt 以上
- Top-5 Jaccard が改善
- 人手スポットチェックで「Top に来る発言の納得感が上がった」と確認できる

## 注意

- DPO は学習データの **質** に超敏感。合成ネガティブだけだと「明らかにダメ」しか学ばないので、必ず人手ペアを混ぜる
- β を大きくしすぎると SFT 性能を破壊する。0.05〜0.2 で振って eval
- KTO や IPO も選択肢。まず DPO で出発、必要なら切替検討

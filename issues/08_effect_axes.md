# #8 [feature] 効果軸(Layer 2)モジュール追加

**Labels**: `enhancement`, `P1`
**Milestone**: v0.3

## 概要

「発言の行為」だけ評価する Layer 1（既存）に加えて、「発言後に議論がどう動いたか」を **自動で**測る Layer 2 を追加。AGENT.md §9 の文脈評価ルールを構造的に活かす層。

## 軸

| 軸 | 判定方法 |
|---|---|
| `focus_convergence` | 後続 K=5 発言の embedding 分散が、その発言の前後で減ったか |
| `decision_reached` | 後続 K ターン以内に「決定」発言が出たか（パターン or LLM） |
| `action_registered` | 「誰が / いつまでに / 何を」が後続 K ターンで確定したか |
| `discussion_diverged` | 後続が脱線方向に動いたか（減点） |

## モジュール構成

```
app/effects/
  __init__.py
  embedding.py          # 日本語埋め込み(モデルは選定後に決定、後述)
  convergence.py        # focus_convergence の計算
  decision_detector.py  # decision_reached / action_registered
  divergence.py         # discussion_diverged の計算
  aggregator.py         # 効果軸を EvaluatedUtterance にマージ
```

## やること

- [ ] `Scores` schema に `EffectScores` を追加 or 別フィールドに切る
  - 互換のため `EvaluatedUtterance.effect_scores: EffectScores | None`
- [ ] 埋め込みは初回ロードしてキャッシュ
- [ ] `decision_detector` は最初パターンマッチ（"決まりました", "やります", "では〜にしましょう"）+ LLM フォールバック
- [ ] `action_registered` は「人名 + 期日 + 動詞」の正規表現 + LLM
- [ ] スコア合算: `total_score = layer1_total + α * effect_total`、α は config で調整
- [ ] `evals/metrics.py` で Layer1 だけ / Layer1+Layer2 / Layer2 だけで精度差を測れるようにする

## 完了条件

- 「短いけど議論を収束させた発言」が Layer2 を有効化したら Top に上がる
- アブレーションスタディが README に追記される

## 注意

`focus_convergence` は発言数が少ない会議でノイズが大きい。`K = min(5, remaining_utterances)` で動的調整。

## 埋め込みモデルの選定

日本語埋め込みは複数候補(多言語汎用 / 日本語特化 / 軽量版)から選ぶ。選定基準:

- 会議発言相当の短文での類似度精度
- 推論速度(CPU で動かす想定なら特に)
- ライセンス・サイズ

この Issue 内で簡易ベンチ(3〜5モデル × 既存サンプル会議)を行い、決定を `docs/embedding_model_selection.md` に残す。**判断 LLM(#17) / 音声(#18) と独立**で扱える小規模選定なので、ここに内包する。

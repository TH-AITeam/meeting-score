# #5 [feature] アノテーションスキーマ定義 + 最初の人手アノテ

**Labels**: `data`, `P0`
**Milestone**: v0.2

## 概要

学習・評価の **正解データ** をどう持つかを決め、3形式すべてを1会議分まず自分で打つ。

## アノテーション3形式

### a) タグ付け (multi-label)

各発言に対して、以下のタグを 0〜多 個つける。speech_type が1ラベルしか持てない制約への補強。

タグ: `論点設定`, `提案`, `深掘り質問`, `情報提供`, `要約`, `リスク提示`, `根拠提示`, `アクション化`, `決定`, `雑談`, `重複`, `脱線`, `上書き`

### b) ペアワイズ比較

同一会議内から 100〜300 ペアサンプリングして「A と B、どちらが会議を前に進めたか」を `A_better` / `B_better` / `tie` で記録。

### c) Top-K / Bottom-K

各会議で「最も貢献した発言 5」「最も無価値な発言 5」を選ぶ。

## スキーマ

```
data/annotations/
  gold/
    v1/
      tags.jsonl              # {meeting_id, utterance_id, tags: [...], annotator}
      pairs.jsonl             # {meeting_id, utt_a, utt_b, winner, annotator}
      top_bottom.jsonl        # {meeting_id, top5: [...], bottom5: [...], annotator}
      meta.json               # アノテータ情報、合意率(κ)、作成日時
```

## やること

- [ ] `data/annotations/README.md` でスキーマ定義
- [ ] アノテ用 CLI `python -m tools.annotate --meeting m001 --mode pairs` を実装
  - ペアをランダムサンプリングして、CLI で選択肢を出す
  - JSONL 追記書き込み
- [ ] サンプル会議 `m001` 全件にタグ付け
- [ ] サンプル会議 `m001` でペアを 100 件アノテ
- [ ] Top5 / Bottom5 をアノテ
- [ ] `evals/metrics.py` の入力データセットがこの形式で読めることを確認

## 完了条件

- `v1` データセットに 1会議分の3形式アノテが入る
- アノテ済みファイルを使って Issue #4 の eval ハーネスが回る

## 注意

- アノテータが1人だけだと過学習する。2人で打って κ 係数も出すのが理想（後追いでもよい）
- 「会議を前進させたか」は **人物の優秀さで判断しない** ように、アノテ UI に注意書きを出す

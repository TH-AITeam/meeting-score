# #12 [training] SFT 用データセット構築パイプライン

**Labels**: `training`, `data`, `P0`
**Milestone**: v0.6

## 概要

ローカル判断モデルを SFT で適合させるためのデータセットを作る。教師信号は **強い API 系モデル(蒸留元)による出力** + **ゴールデン人手アノテ** のハイブリッド。

蒸留元モデルは別途決定する(プロンプト品質 / 価格 / 規約で選ぶ)。採用結果は `data/sft/README.md` に明記する。

## データ形式

採用ベースモデル(#17)の chat template に合わせた JSONL。

```json
{
  "messages": [
    {"role": "system", "content": "あなたは会議分析アシスタントです..."},
    {"role": "user", "content": "<同じプロンプト本体>"},
    {"role": "assistant", "content": "<JSON のみ>"}
  ],
  "meta": {"source": "gold|distilled", "meeting_id": "m001", "utterance_id": "u003"}
}
```

## データソース3層

| Tier | 規模 | 教師信号 | 目的 |
|---|---|---|---|
| gold | 500〜1500 | 人手アノテ(#5) | 軸採点の正解 |
| distilled | 5000〜20000 | 蒸留元モデルの出力 | プロンプト最適解の蒸留 |
| synthetic | 1000〜5000 | #7 で生成 | エッジケース補強 |

## やること

- [ ] `scripts/build_sft_dataset.py`
  - input: `data/annotations/gold/v1/*` + 蒸留対象会議
  - output: `data/sft/v1/{train,val,test}.jsonl`
  - 蒸留時は採用した蒸留元モデルに同じプロンプトで叩いて出力を保存
  - reject ループ: JSON Schema 違反 / 合計点が極端 / reason が空のものは捨てる
- [ ] train/val/test 分割は **会議単位** で(発言単位だとリーク)
- [ ] chat template は採用ベースモデル(#17)に合わせる
- [ ] `data/sft/README.md` にライセンス・出典・蒸留時の元モデルを必ず記載

## 完了条件

- `train.jsonl` が 5000 件以上
- 全件で JSON Schema バリデーション通過
- `val.jsonl` の会議が `train.jsonl` に含まれない
- 統計サマリ(軸ごとのスコア分布、speech_type 分布)が `docs/sft_data_stats.md` に出る

## 注意

- 蒸留元モデルの利用規約を確認(API 提供元の Terms)
- 商用展開する場合は蒸留データの扱いが特に問題になる。今は研究用途と明記する
- 蒸留元と採用ベースモデル(#17)が同系統(同社製)だと比較が偏るので、できるだけ系統を変える

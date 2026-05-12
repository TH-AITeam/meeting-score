# data/annotations: 人手アノテーション

会議貢献度スコアの **正解データ** を JSONL 形式で保持する。Issue #5 (eval ハーネス) と
Issue #6 (アノテ収集) で共有するスキーマ。

## ディレクトリ構成

```
data/annotations/
├── README.md           # このファイル
└── gold/               # 評価用ゴールドアノテ
    └── v1/             # バージョン管理（v2 以降は別ディレクトリで追記）
        ├── tags.jsonl
        ├── pairs.jsonl
        ├── top_bottom.jsonl
        ├── meetings/   # 評価対象会議 JSON（{meeting_id}.json）
        └── meta.json   # アノテータ情報・κ 係数・作成日時
```

- バージョン: スキーマ・分布を変えたら v2/, v3/ ... を追加する。eval ハーネスは
  `--dataset data/annotations/gold/v1` のようにディレクトリを丸ごと指定する。

## ファイル形式

すべて **JSONL**（1行=1レコード）。先頭が `#` の行と空行はコメントとして無視する。

### tags.jsonl — タグ付け（multi-label）

`speech_type` 単一ラベルでは捉えきれない発言の性質をタグで補強する。

```jsonl
{"meeting_id": "m001", "utterance_id": "u003", "tags": ["論点設定", "要約"], "annotator": "akiyama"}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `meeting_id` | string | 会議 ID |
| `utterance_id` | string | 発言 ID |
| `tags` | string[] | 下記の許容タグから 0 個以上 |
| `annotator` | string | アノテータ識別子（既定: `unknown`） |

許容タグ:
`論点設定`, `提案`, `深掘り質問`, `情報提供`, `要約`, `リスク提示`,
`根拠提示`, `アクション化`, `決定`, `雑談`, `重複`, `脱線`, `上書き`

### pairs.jsonl — ペアワイズ比較

同一会議内の2発言 `utt_a` / `utt_b` のうちどちらが会議を前進させたかを記録する。
人物の優秀さではなく **発言の貢献** で判断すること。

```jsonl
{"meeting_id": "m001", "utt_a": "u003", "utt_b": "u007", "winner": "A_better", "annotator": "akiyama"}
{"meeting_id": "m001", "utt_a": "u010", "utt_b": "u012", "winner": "tie", "annotator": "akiyama"}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `meeting_id` | string | 会議 ID |
| `utt_a` / `utt_b` | string | 比較する発言 ID（同一会議内） |
| `winner` | `"A_better"` \| `"B_better"` \| `"tie"` | 勝者ラベル |
| `annotator` | string | アノテータ識別子 |

eval ハーネスはスコア差が `tie_threshold`（既定 0.5）以下なら tie と判定する。

### top_bottom.jsonl — Top-K / Bottom-K

1会議につき1行。最も貢献した発言5件と、最も無価値な発言5件の `utterance_id` を列挙する。

```jsonl
{"meeting_id": "m001", "top5": ["u003","u015","u020","u021","u022"], "bottom5": ["u005","u006","u008","u011","u017"], "annotator": "akiyama"}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `meeting_id` | string | 会議 ID |
| `top5` | string[] | 上位 5 発言の ID（順序は重要視しない） |
| `bottom5` | string[] | 下位 5 発言の ID |
| `annotator` | string | アノテータ識別子 |

### meta.json — メタ情報（任意）

```json
{
  "version": "v1",
  "created_at": "2026-05-12",
  "annotators": ["akiyama", "wyvern"],
  "kappa": {
    "tags": 0.72,
    "pairwise": 0.65,
    "top_bottom": 0.58
  },
  "notes": "...任意の補足..."
}
```

## eval ハーネスとの対応

eval ハーネスのメトリクスは以下のように対応する:

| アノテ形式 | 使われるメトリクス |
|---|---|
| `pairs.jsonl` | Pairwise accuracy、Spearman / Kendall tau（疑似ランクの基礎） |
| `top_bottom.jsonl` | Top5 Jaccard、Bottom5 Jaccard |
| `tags.jsonl` | （v1 では分析・誤評価分類用。将来 F1 系メトリクスで使う） |

実行例:

```bash
task eval DATASET=../data/annotations/gold/v1
```

会議元データは `gold/v1/meetings/{meeting_id}.json` を既定として探し、
無ければ `data/sample_meetings/` をフォールバックとして読む。

## バリデーション

`evals.schema.load_*_annotations` が Pydantic で検証する:

- `winner` は 3 値の Literal
- `tags` は文字列リスト（許容タグの妥当性は warn ベース、現状は緩い検証）
- JSON パース失敗・型不一致は `ValueError` で行番号付きで報告

## バージョニング

スキーマ破壊変更（フィールド名変更・必須化など）が発生したら v1/ をフリーズして v2/ を切る。
追加カラム（任意フィールド）は v1 内で行ってよい。

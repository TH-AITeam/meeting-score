# 国会会議録 → LoRA 学習データ（蒸留）

`data/kokkai/pages/` の国会会議録（生データ・ラベルなし）から、
発言評価モデル（入力発言 → コメント＋スコア）を LoRA 微調整するための
**蒸留（distillation）** データを作る手順とプロンプトを置く。

教師モデル（OpenAI 等の高性能 LLM）に既存の評価ロジックと同じ採点をさせ、
その入出力ペアを学習データにして、ローカルの小型モデルへ LoRA で蒸留する。

## なぜ蒸留が必要か

国会会議録には `scores` も `reason` も無く、LoRA（教師あり微調整）に必須の
**入力→出力ペア**が存在しない。さらに評価プロンプトが要求する
`goal / agenda / decision_points / current_topic` も含まれない。
そこで以下の 2 段で教師ラベルを生成する。

```
pages/*.json  ──①メタ抽出──▶  会議メタ(goal/agenda/…)
     │                                  │
     └──②各発言＋前後文脈＋メタ──▶ 教師LLM ──▶ {speech_type, scores, penalties, reason}
                                                          │
                                                          ▼
                                       LoRA 学習用 JSONL（messages 形式）
```

## 推奨する学習形式：会話形式 JSONL（messages）

1 発言 = 1 サンプル。axolotl / LLaMA-Factory / trl / Unsloth がそのまま読める。

```json
{"messages": [
  {"role": "user", "content": "<本番の評価プロンプトそのまま>"},
  {"role": "assistant", "content": "{\"speech_type\":\"懸念提示\",\"scores\":{...},\"penalties\":{...},\"reason\":\"...\"}"}
]}
```

### 最重要原則：train と inference のプロンプトを一致させる
- `user` フィールドは **本番推論で使う `backend/app/evaluators/prompt.py::build_prompt()` の出力をそのまま**入れる
  （= `backend/prompts/utterance_eval.txt` にメタ・前後文脈・対象発言を埋めたもの）。
- 学習プロンプトが本番とズレると LoRA の効果が激減する。
- このディレクトリの `02_utterance_eval_teacher.txt` は **教師ラベルを生成するためだけ**のプロンプト
  （国会向けの追加指針を含む）。学習サンプルの `user` には使わない。役割を混同しないこと。

### assistant（ラベル）は本番スキーマ準拠
`prompt.py::RESPONSE_SCHEMA` と完全一致させる。生成後に
`normalize_result()` を通して値域（scores 0〜3 / penalties -3〜0）と
`speech_type` を正規化してから書き出すと、本番パーサと齟齬が出ない。

## プロンプト一覧

| ファイル | 用途 | 主なプレースホルダ |
|---|---|---|
| `prompts/01_meta_extraction.txt` | 会議ごとに goal/agenda/decision_points/meeting_type を抽出 | `$name_of_meeting`, `$name_of_house`, `$date`, `$transcript` |
| `prompts/02_utterance_eval_teacher.txt` | 1 発言の教師ラベルを生成（国会向け追加指針あり） | `build_prompt()` と同じ（`$meeting_type`, `$meeting_goal`, `$agenda`, `$decision_points`, `$current_topic`, `$before_utterances`, `$target_speaker`, `$target_timestamp`, `$target_text`, `$after_utterances`） |

## データ生成手順（実装時の流れ）

1. `pages/*.json` を読み、`meetingRecord[].speechRecord[]` を会議（issueID）ごとに分割。
2. **発言フィルタ**（学習ノイズ除去）。次は除外推奨：
   - `speaker == "会議録情報"`、出席者名簿・資料貼り付けの塊
   - 「異議なし」「挙手多数。よって、そのように決定いたしました」等の純定型進行
   - 極端に短い相づち・氏名のみの指名（「○○君。」）
   - 残すべき: 討論・質疑・趣旨説明・賛成/反対意見など実質発言。
3. 会議全文を `01_meta_extraction.txt` に渡してメタを取得（会議内でキャッシュ・1 会議 1 回）。
4. 各対象発言について、前後 N 件（本番デフォルト 3）の文脈を組み立て、
   `02_..._teacher.txt` で教師ラベル JSON を生成 → `normalize_result()` で正規化。
5. 学習サンプルを書き出す。`user` は **本番 `build_prompt()` の出力**、`assistant` は正規化済み JSON 文字列。
6. `train.jsonl` / `val.jsonl` に分割（会議単位で分けてリーク防止）。

## 実装スクリプト（この手順を半自動化）

上記 1〜6 を 3 段に分けて実装済み。教師ラベル生成（メタ抽出・発言採点）は
Claude 本体／サブエージェントが上記 2 プロンプトに従って担う。

| ステップ | スクリプト | 入出力 |
|---|---|---|
| ①機械処理 | `scripts/lora/build_distill_jobs.py` | 生 JSONL → `jobs/<issueID>.json`（ノイズ除去・前後文脈整形済みの採点ジョブ） |
| ②ラベル付け | （教師 LLM が担当） | `jobs/*.json` + プロンプト01/02 → `labels/<issueID>.json`（meta + 各発言ラベル） |
| ③機械処理 | `scripts/lora/assemble_distill_data.py` | `jobs/` + `labels/` → `train.jsonl` / `val.jsonl`（messages 形式・会議単位分割） |

```bash
# ①パイロット抽出（多様な委員会を 10 会議・前後文脈 N=3）
python scripts/lora/build_distill_jobs.py --max-meetings 10 --context 3
# ②各 jobs/*.json をプロンプト01→02 でラベル付けし labels/<issueID>.json に保存
# ③回収して train/val を生成（会議単位で val 2 割）
python scripts/lora/assemble_distill_data.py --val-ratio 0.2
```

`assemble_distill_data.py` は `backend/prompts/utterance_eval.txt` と
`_MEETING_TYPE_LABELS` を読み込み、本番 `build_prompt()` と同一の `user` を
再現する。`current_topic` は会議録に逐次トピックが無いため本番同様 `(未設定)`
で統一。ラベルは本番 `normalize_result` 相当の値域クランプを通して書き出す。

## 出力先（推奨）
```
data/kokkai/distill/
  prompts/                 # 本ディレクトリ（プロンプト）
  meta/<issueID>.json      # 抽出した会議メタ（再利用・キャッシュ）
  labels/<issueID>.jsonl   # 発言ごとの教師ラベル（中間生成物）
  train.jsonl / val.jsonl  # 最終 LoRA 学習データ（messages 形式）
```

## 量・品質の目安
- まず数百〜数千発言で試作し、val でスコア相関（教師 vs 生徒）と reason の質を確認。
- 教師の reason は 1〜2 文に収め、JSON 以外を吐かせない（`02` の出力指示を厳守）。
- `decision`（議事決定）会議に偏るため、討論の多い委員会（予算・厚労・内閣など）を混ぜて軸の分布を確保する。

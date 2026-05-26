# data/sft: SFT 用データセット (Issue #13)

ローカル判断モデルを SFT で適合させる学習データ。`scripts/build_sft_dataset.py`
が生成する。**研究用途**であり、商用展開時は蒸留データの扱い（蒸留元 API の規約）を
別途確認すること。

## 形式

`v1/{train,val,test}.jsonl`。1 行 = 1 発言。本番推論に合わせ `user` +
`assistant` のみ（system ロールは持たない）。

```json
{"messages": [
   {"role": "user", "content": "<本番 build_prompt() と同一のプロンプト>"},
   {"role": "assistant", "content": "{\"speech_type\":...,\"scores\":{...},\"penalties\":{...},\"reason\":\"...\"}"}
 ],
 "meta": {"source": "gold|distilled", "meeting_id": "...", "utterance_id": "..."}}
```

- `user` は `backend/prompts/utterance_eval.txt`（本番 `build_prompt()`）をそのまま使う。
  **train と inference のプロンプトを一致させること**（ズレると SFT/LoRA の効果が激減する）。
- `assistant` は本番 `prompt.normalize_result` 準拠（scores 0〜3 / penalties -3〜0 / speech_type 正規化）。
- chat template は採用ベースモデル (#17) に合わせて学習側で適用する。

## データソース 3 層

| Tier | 教師信号 | 状態 |
|---|---|---|
| distilled | 教師 LLM の出力（`data/annotations/kokkai/distill/`） | 利用可 |
| gold | 人手アノテ (#5) | **保留**。現行 gold アノテ (tags/pairs/top_bottom) は eval ハーネス用で、SFT の `assistant`（軸採点 JSON）形式ではない。#5/#6 が軸採点ラベルを出したら `--gold-dir` 配下に distill と同じ jobs/+labels/ 構造で投入する |
| synthetic | #7 で生成 | 未実装 |

## 蒸留元（教師）モデル

- **Claude (本体/サブエージェントが prompts 01/02 に従い手動ラベリング)**
- 蒸留元と採用ベースモデル (#17) が同系統だと比較が偏るため、できるだけ系統を変えること。
- 蒸留元 API の利用規約（Terms）を確認すること。

## 分割

train/val/test は **会議単位**で分割しリークを防ぐ（同一会議の発言が複数 split に跨らない）。

## 再生成

```bash
python scripts/build_sft_dataset.py --out-dir data/sft/v1 --val-ratio 0.15 --test-ratio 0.15
```

統計サマリは `docs/sft_data_stats.md` に出力される。

## 完了条件 (#13) と現状

- [ ] `train.jsonl` >= 5000 件 — **教師ラベリング / gold 収集（データ収集タスク）に依存**。
  現状 distill のラベル済み会議数に比例。会議を増やせば同コマンドで規模拡大できる。
- [x] 全件 JSON Schema バリデーション通過（`assistant` を `RESPONSE_SCHEMA` で検証）
- [x] `val`/`test` の会議が `train` に含まれない（会議単位分割）
- [x] 統計サマリ（軸スコア分布・speech_type 分布）を `docs/sft_data_stats.md` に出力

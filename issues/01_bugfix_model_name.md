# #1 [bug] `gpt-5.4-mini` という実在しないモデル名を直す

**Labels**: `bug`, `P0`
**Milestone**: v0.2

## 概要

`config.yaml`, `app/scoring/weights.py`, `app/evaluators/llm_evaluator.py` の3箇所に `gpt-5.4-mini` が書かれているが、これは実在しないモデル名。OpenAI API を叩いた瞬間に 404 で落ちる。

## 影響

- 現状、本番実行で必ず失敗する
- リトライ3回後にデフォルト値(全項目0)が返るので、見た目は動いてしまう ← より悪い

## やること

- [ ] `config.yaml:23` の `llm.model` を、現時点で OpenAI API に存在する有効なモデル名に修正
- [ ] `app/scoring/weights.py:34, 70` のデフォルト値も合わせて修正
- [ ] `app/evaluators/llm_evaluator.py:214` の関数デフォルト引数も修正
- [ ] 修正後に `data/sample_meetings/sample_meeting_01.json` で smoke test

## 完了条件

- サンプル会議1本で End-to-End が落ちずに完走する
- すべての発言で `evaluation_failed=False` が返る

## 補足

これは **v0.5 でローカルモデル(#17)に置き換わるまでの暫定対処**。 採用モデルを Issue で決めるのは #17 側で行うので、ここでは「動けばいい」レベルで OK。

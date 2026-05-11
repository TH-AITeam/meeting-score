# #3 [bug] `_estimate_current_topic` の議題推定がガバい

**Labels**: `bug`, `P1`
**Milestone**: v0.2

## 概要

`app/context_builder/builder.py:23` の `_estimate_current_topic` は、発言を「アジェンダ項目数で等分」して現在議題を推定している。会議の進行が等分されることはないので、ほぼ常にハズれる。LLM プロンプトの `current_topic` に嘘の値が入って評価精度を落とす。

## やること

- [ ] `_estimate_current_topic` のフォールバックを廃止 or 空文字を返すように変更
- [ ] `current_topic` が空のときは、プロンプト側で「(議題不明)」と扱う
- [ ] `topic_transitions` が指定されている場合のみ `current_topic` を埋める方針に統一
- [ ] `tests/test_context_builder.py` でフォールバック動作を検証

## やらないこと（別 Issue）

- LLM で議題区切りを自動検出する。 → 将来の改善。今は `topic_transitions` を人が打つ前提で十分。

## 完了条件

- `topic_transitions` 未指定の会議で、`current_topic` が「等分割で推定された嘘」を返さないこと
- 既存のサンプル会議は `topic_transitions` が入っているので動作不変であること

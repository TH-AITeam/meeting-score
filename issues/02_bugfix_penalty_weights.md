# #2 [bug] `config.yaml` の `penalties:` セクションが読み込まれていない

**Labels**: `bug`, `P0`
**Milestone**: v0.2

## 概要

`config.yaml` には以下のセクションがあるが、

```yaml
penalties:
  duplication: 1.0
  verbosity: 1.0
  off_topic: 1.0
  unsupported_assertion: 1.0
```

`app/scoring/weights.py` の `load_config` はこれを完全に無視。`ScoringWeights` dataclass にも対応フィールドがなく、`app/scoring/calculator.py` も penalty を素のまま足している。設定書いた人が「重み変えてるつもり」になっているサイレント失敗。

## やること

- [ ] `PenaltyWeights` dataclass を `app/scoring/weights.py` に追加
- [ ] `AppConfig` に `penalty_weights: PenaltyWeights` を持たせる
- [ ] `load_config` で `penalties:` セクションを読む
- [ ] `app/scoring/calculator.py:calculate_total_score` で penalty に重みを掛ける
- [ ] `tests/test_scoring.py` に penalty 重み変更時のテストを追加
- [ ] `tests/test_rule_corrections.py` も連動修正

## 完了条件

- `config.yaml` で `penalties.duplication: 2.0` にすると、重複検出時に従来の倍の減点になることをテストで検証

## 注意

penalty は元々負の値（0〜-3）なので、重みは正の倍率として掛ける。

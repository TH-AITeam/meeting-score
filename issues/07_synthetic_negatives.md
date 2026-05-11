# #7 [data] 合成ネガティブ生成スクリプト

**Labels**: `data`, `P1`
**Milestone**: v0.2

## 概要

ゴールデンの良発言を LLM で「劣化」させてポジ／ネガペアを大量生成する。これが DPO 学習(#14) と eval プロンプト調整に効く。

## 劣化パターン

| パターン | 操作 |
|---|---|
| **冗長化** | 同じ内容を1.5〜3倍の文字数に水増し |
| **根拠剥がし** | 数字・固有名詞・条件を消して断定だけ残す |
| **脱線挿入** | 文末に別話題への遷移を1文足す |
| **重複化** | 直前の他者発言とほぼ同じ内容に書き換える |
| **マウント化** | 直前発言を無視して、自説に塗り替える |

## ディレクトリ構成

```
scripts/
  generate_synthetic_negatives.py
data/annotations/
  synthetic/
    v1/
      pairs.jsonl           # {meeting_id, utt_a, utt_b, winner: "A", pattern: "verbose"}
      degraded_utterances.jsonl  # 元発言 + 劣化版発言
```

## やること

- [ ] `scripts/generate_synthetic_negatives.py` 実装
  - 入力: ゴールデン Top-K 発言
  - 各パターンで強い LLM (#17 で採用したモデル or 蒸留用 API) に劣化を依頼
  - JSON 出力（強制スキーマ）
- [ ] 出力検証: 劣化版が元より明らかに悪いかを別 LLM で自動チェック（reject ループ）
- [ ] 1000〜5000 ペア生成
- [ ] eval ハーネス(#4)に追加データセットとして食わせる

## 完了条件

- `synthetic/v1` に 1000ペア以上
- ペアワイズ accuracy が gold だけのとき vs gold + synthetic のときで差分が取れる

## 注意

- 劣化が極端すぎると「明らかにダメ」ばかり学んでしまい、現場の微妙な差で効かなくなる。**温度を 0.3〜0.7 で振る**
- 同一元発言から複数パターン作って、パターン間の優劣も使えるようにする

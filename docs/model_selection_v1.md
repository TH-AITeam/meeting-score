# 判断モデル選定 v1 (Issue #18)

候補比較の総合スコアシート。実測値は SSH 先 (RTX 5090 32GB) で
`scripts/run_model_benchmark.sh --all` を回した結果を転記する。

| 環境項目 | 値 |
|---|---|
| 実行ホスト | SSH 先 RTX 5090 (32GB VRAM, sm_120) |
| 推論サーバ | vLLM 0.6+ (`xgrammar` guided JSON) |
| 評価データ | `data/annotations/gold/v1`（#6 完了後に固定）+ `data/sample_meetings/sample_meeting_01.json` |
| 評価日 | TBD（#6 完了後に更新） |

## スコアシート

| モデル | 量子化 | VRAM (GB) | Spearman | Kendall τ | Top5 Jaccard | Pairwise acc | JSON 成功率 | mean SD (7軸) | p50 (ms) | p95 (ms) | ライセンス | 学習しやすさ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B-Instruct | AWQ INT4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Apache 2.0 | ★★★ |
| Qwen3-14B-Instruct | bf16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Apache 2.0 | ★★★ |
| Llama-3.3-8B-Instruct | bf16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Llama 3 CL | ★★★ |
| Swallow-Llama-3.1-8B-Instruct | bf16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Llama 3.1 CL + Swallow | ★★☆ |
| Phi-4-14B | bf16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | MIT | ★★★ |

数値の出典: `reports/model_benchmarks/{served_name}/{timestamp}*.json`

## 軸重み（採用判定用）

| 軸 | 重み |
|---|---|
| Spearman + Kendall τ（日本語ランキング精度） | **0.40** |
| Pairwise accuracy | 0.20 |
| Top5 Jaccard | 0.10 |
| JSON 成功率 | 0.10 |
| mean SD（小さいほど良） | 0.10 |
| p95 レイテンシ（小さいほど良） | 0.05 |
| ライセンス自由度 | 0.05 |

実測が揃った段階で min-max 正規化 → 重み付き合計でランキングを出す。

## 暫定採用（実測前の予想）

実測値が出るまでの暫定判断（公開ベンチ + 一般的な評判から）:

- **第一採用候補**: **Qwen3.6-27B-Instruct (AWQ INT4)**
  - 同シリーズの 14B/30B-MoE が JMT-Bench / Nejumi で上位定着
  - AWQ で 14GB に収まり、コンテキスト 16K+ を残せる
  - Apache 2.0 で SFT/DPO/蒸留に制約なし
- **第二採用候補（控え）**: **Qwen3-14B-Instruct (bf16)**
  - 27B が SFT で重い場合の controller / draft 用
  - 推論速度は 27B の 2 倍前後を想定
- **対照群（採用しない前提のベースライン）**:
  - Llama-3.3-8B: 英語ベースの 8B 下限
  - Swallow-3.1-8B: 日本語特化 8B との比較
  - Phi-4-14B: 同サイズ別系統の比較

**最終採用は実測後に確定し、本ドキュメントを上書きする。**

## 実測手順

SSH 先で:

```bash
# リポジトリ clone & 依存セットアップ
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score
uv sync
uv pip install vllm

# 全候補を順に回す
bash scripts/run_model_benchmark.sh --all
```

結果は `reports/model_benchmarks/` 配下に出る。Mac に rsync で持ち帰り、
本ドキュメントの表に転記する。

## 採用後にやること（受け入れ条件のチェックリスト）

- [ ] 採用モデル名を `backend/config.yaml.example` の `llm.model` に書き込む
- [ ] `docs/adr/0001-judgment-model.md` の Status を `Proposed` → `Accepted` に変更
- [ ] `docs/model_history.md` に v1 採用エントリを追加
- [ ] Issue #11 / #13 / #14 / #16 のコメントで採用モデル名を共有

## 関連

- Issue #18 (本 Issue)
- Issue #5 / `backend/evals/`: メトリクス算出基盤
- Issue #12 / `docs/inference_server_selection.md`: vLLM 採用根拠
- `docs/model_candidates.md`: 候補絞り込み詳細
- `docs/adr/0001-judgment-model.md`: 意思決定の根拠記録

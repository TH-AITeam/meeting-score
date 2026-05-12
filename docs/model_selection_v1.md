# 判断モデル選定 v1 (Issue #18)

候補比較の総合スコアシート。実測値は SSH 先 (RTX 5090 32GB) で
`scripts/run_model_benchmark.sh --all` を回した結果を転記する。

| 環境項目 | 値 |
|---|---|
| 実行ホスト | SSH 先 RTX 5090 (32GB VRAM, sm_120) |
| 推論サーバ | vLLM 0.20.x (xgrammar 既定の JSON 強制) |
| 評価データ | `data/annotations/gold/v1`（#6 完了後に固定）+ `data/sample_meetings/sample_meeting_01.json` |
| 評価日 | TBD（実測完了後に更新） |

## スコアシート

| モデル | 量子化 | VRAM (GB) | Spearman | Kendall τ | Top5 Jaccard | Pairwise acc | JSON 成功率 | mean SD (7軸) | p50 (ms) | p95 (ms) | ライセンス | 学習しやすさ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Apache 2.0 | ★★★ |
| Qwen3-14B | bf16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Apache 2.0 | ★★★ |
| Qwen2.5-32B-Instruct-AWQ | AWQ INT4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Apache 2.0 | ★★★ |
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

- **第一採用候補**: **`Qwen/Qwen3.6-27B`**（BnB NF4 でオンザフライ 4bit 量子化）
  - 最新世代の Qwen で日本語ベンチ上位を維持する見込み
  - 公式 AWQ チェックポイントが現状未公開のため BnB を使う一時運用。公式 AWQ 公開時に置き換え
- **第二採用候補（控え）**: **`Qwen/Qwen3-14B`** (bf16)
  - 27B が SFT で重い場合の controller / draft 用
  - 同系統で tokenizer / chat template を共有可能
- **対照群（採用しない前提のベースライン）**:
  - Qwen2.5-32B-Instruct-AWQ: 世代間比較（Qwen2.5 vs Qwen3）
  - Swallow-3.1-8B: サイズ最小 + 日本語特化の代替ライン
  - Phi-4-14B: 同サイズ別系統の比較

**最終採用は実測後に確定し、本ドキュメントを上書きする。**

## 実測手順

SSH 先で:

```bash
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score
uv venv --python 3.13 --clear  # Python 3.13 必須 (pyproject.toml の要件)
source backend/.venv/bin/activate
cd backend && uv sync && cd ..
uv pip install vllm

# 全候補を順に回す（最大 30 分 × 5 = 2.5 時間 + 推論時間）
bash scripts/run_model_benchmark.sh --all 2>&1 | tee /tmp/issue18_bench.log
```

結果は `reports/model_benchmarks/` 配下に出る。Mac に rsync で持ち帰り、本ドキュメントの表に転記する。

## 採用後にやること（受け入れ条件のチェックリスト）

- [ ] 採用モデル名を `backend/config.yaml.example` の `llm.model` に書き込む
- [ ] `docs/adr/0001-judgment-model.md` の Status を `Proposed` → `Accepted` に変更
- [ ] `docs/model_history.md` に v1 採用エントリを追加（採用日を確定）
- [ ] Issue #11 / #13 / #14 / #16 のコメントで採用モデル名を共有

## 関連

- Issue #18 (本 Issue)
- Issue #5 / `backend/evals/`: メトリクス算出基盤
- Issue #12 / `docs/inference_server_selection.md`: vLLM 採用根拠
- `docs/model_candidates.md`: 候補絞り込み詳細
- `docs/adr/0001-judgment-model.md`: 意思決定の根拠記録

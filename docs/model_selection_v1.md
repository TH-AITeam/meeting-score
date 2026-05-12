# 判断モデル選定 v1 (Issue #18)

候補比較の総合スコアシート。SSH 先 RTX 5090 (32GB / Blackwell) で
`scripts/run_model_benchmark.sh --all` を実行した実測値を反映している。

| 環境項目 | 値 |
|---|---|
| 実行ホスト | SSH 先 RTX 5090 (32GB VRAM, sm_120 / Blackwell) |
| 推論サーバ | vLLM 0.20.2 (xgrammar 既定の JSON 強制) |
| Attention backend | FLASH_ATTN (FlashInfer は Blackwell 未対応で除外) |
| KV cache dtype | fp8 |
| 評価データ (latency / stability) | `data/sample_meetings/sample_meeting_01.json` |
| 評価データ (Spearman / pairwise) | **TBD** (Issue #6 のゴールド v1 完成後に固定) |
| 評価日 (latency / stability) | 2026-05-12 |

## スコアシート

精度系 (Spearman / Kendall τ / Top5 Jaccard / Pairwise) は **Issue #6 のゴールデンアノテーション完成後** に埋める。本表ではレイテンシ / JSON 成功率 / 安定性 (greedy decode) を中心に比較した。

| モデル | 量子化 | Spearman | Kendall τ | Top5 Jaccard | Pairwise acc | JSON 成功率 | mean SD (7軸) | p50 (ms) | p95 (ms) | ライセンス | 学習しやすさ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B (unsloth NVFP4)** | NVFP4 (compressed-tensors) | TBD | TBD | TBD | TBD | **100.0%** | 0.000 | **7,420** | 7,487 | Apache 2.0 | ★★★ |
| Qwen2.5-32B-Instruct-AWQ | AWQ INT4 (Marlin) | TBD | TBD | TBD | TBD | 100.0% | 0.000 | **2,973** | 2,987 | Apache 2.0 | ★★★ |
| Qwen3.6-27B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | 100.0% | 0.000 | 9,783 | 9,812 | Apache 2.0 | ★★★ |
| Qwen3-14B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | 100.0% | 0.010 | 3,360 | 3,407 | Apache 2.0 | ★★★ |
| Phi-4-14B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | 100.0% | 0.022 | 3,732 | 3,748 | MIT | ★★★ |
| Swallow-Llama-3.1-8B-Instruct | bf16 | TBD | TBD | TBD | TBD | 100.0% | 0.035 | 1,513 | 1,515 | Llama 3.1 CL + Swallow | ★★☆ |

数値の出典: `reports/model_benchmarks/{served_name}/20260512_*_latency.json` および `_stability.json`

### 観察

- **JSON 成功率は全モデル 100%**。vLLM の xgrammar guided JSON が全モデルで安定動作する
- **安定性 (mean SD)**: ほぼ 0 (greedy decode のため決定的)。Phi-4 / Swallow は内部に若干の非決定性あり。真の安定性測定には `temperature > 0` が必要 (`LocalEvaluator` 拡張、別 Issue)
- **レイテンシは量子化形式の差が支配的**:
  - AWQ Marlin (3.0s) ≪ NVFP4 (7.4s) ≪ BnB NF4 (9.8s)
  - 同じ Qwen 3 世代でも、Qwen2.5-32B-AWQ < Qwen3.6-27B-BnB と量子化で逆転
- **MoE の効果**: Qwen3.6-35B-A3B は 35B 総 / 3B アクティブだが、現状の NVFP4 実装ではフル dense 32B-AWQ より遅い。今後 vLLM の FP4 / MoE 推論最適化が進めば改善見込み

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

## 採用判断（暫定、Proposed）

- **第一採用候補**: **`unsloth/Qwen3.6-35B-A3B-NVFP4`**
  - 最新 Qwen3.6 世代の MoE モデル (35B 総 / 3B アクティブ)
  - Blackwell ネイティブの NVFP4 量子化、5090 32GB に余裕で乗る
  - Apache 2.0 で SFT/DPO/蒸留に制約なし
  - レイテンシは AWQ Marlin (32B) に劣るが、知識量と将来性で勝つ判断
- **控え採用**: **`Qwen/Qwen2.5-32B-Instruct-AWQ`**
  - レイテンシ最速 (p50 3.0s)、JSON 安定、ライセンスフリー
  - Qwen3.6 系で精度が思ったほど出ない場合の即時切替先
- **対照群 (採用しない)**:
  - Qwen3.6-27B BnB: BnB のオーバーヘッドでレイテンシ最遅 (9.8s)
  - Qwen3-14B / Phi-4-14B BnB: サイズダウン版の参考
  - Swallow-3.1-8B: 日本語特化の下限ライン

**Status は `Proposed`**。Spearman / pairwise の精度測定が Issue #6 完成後に実施され、結果次第で第一採用を Qwen2.5-32B-AWQ や別候補に切り替える可能性がある。

## 実測手順

SSH 先 (RTX 5090) で:

```bash
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score

# venv (Python 3.13 必須)
cd backend && uv venv --python 3.13 --clear && cd ..
source backend/.venv/bin/activate
cd backend && uv sync && cd ..

# vLLM + bitsandbytes
uv pip install vllm bitsandbytes

# Blackwell では FlashInfer JIT が落ちるので外す
uv pip uninstall flashinfer-python

# 全候補を順に回す (3〜4 時間)
bash scripts/run_model_benchmark.sh --all 2>&1 | tee /tmp/issue18_bench.log

# 集計
python scripts/aggregate_benchmark_results.py --out reports/model_benchmarks/_summary.md
```

結果は `reports/model_benchmarks/{served_name}/{ts}_*.json` に出る。

## 採用後にやること（受け入れ条件のチェックリスト）

- [x] 採用モデル名を `backend/config.yaml.example` の `llm.model` に書き込む
- [ ] Issue #6 (アノテ) 完成後、`make eval` で Spearman / pairwise を実測 → 本表の TBD 列を埋める
- [ ] 精度差を見て ADR の Decision / Status を Accepted へ更新（または採用モデル変更）
- [ ] `docs/model_history.md` の v1 採用日を最終確定日に更新
- [ ] Issue #11 / #13 / #14 / #17 のコメントで採用モデル名を共有

## 関連

- Issue #18 (本 Issue)
- Issue #5 / `backend/evals/`: メトリクス算出基盤
- Issue #12 / `docs/inference_server_selection.md`: vLLM 採用根拠
- `docs/model_candidates.md`: 候補絞り込み詳細
- `docs/adr/0001-judgment-model.md`: 意思決定の根拠記録

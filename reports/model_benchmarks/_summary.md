# モデル比較ベンチマーク集計結果

reports_dir: `reports/model_benchmarks`

対象モデル数: 6

集計時刻ベース (各モデル最新): phi-4-14b-bnb=20260512_101603, qwen2.5-32b-awq=20260512_081247, qwen3-14b-bnb=20260512_075906, qwen3.6-27b-bnb=20260512_071952, qwen3.6-35b-nvfp4=20260512_114948, swallow-3.1-8b-bf16=20260512_083657

## スコアシート

| モデル | 量子化 | VRAM (GB) | Spearman | Kendall τ | Top5 Jaccard | Pairwise acc | JSON 成功率 | mean SD (7軸) | p50 (ms) | p95 (ms) | ライセンス | 学習しやすさ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.000 | 9783 | 9812 | Apache 2.0 | ★★★ |
| Qwen3-14B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.010 | 3360 | 3407 | Apache 2.0 | ★★★ |
| Qwen2.5-32B-Instruct-AWQ | AWQ INT4 | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.000 | 2973 | 2987 | Apache 2.0 | ★★★ |
| Swallow-Llama-3.1-8B-Instruct | bf16 | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.035 | 1513 | 1515 | Llama 3.1 CL + Swallow | ★★☆ |
| Phi-4-14B | BnB NF4 (on-the-fly) | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.022 | 3732 | 3748 | MIT | ★★★ |
| qwen3.6-35b-nvfp4 | ? | TBD | TBD | TBD | TBD | TBD | 100.0% | 0.000 | 7420 | 7487 | ? | ? |

## 反映手順

1. 上記表を `docs/model_selection_v1.md` の同等表に置き換える
2. `docs/adr/0001-judgment-model.md` の Status を `Proposed` → `Accepted` に更新
3. `docs/model_history.md` の v1 行の採用日を確定日に更新

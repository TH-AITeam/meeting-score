# 判断モデル候補リスト (Issue #18)

会議貢献度スコアリングの判断 LLM 候補。**Mac で開発・SSH 越し RTX 5090 (32GB VRAM, Blackwell sm_120) で実行** を前提に絞り込んだ 6 モデル。HuggingFace 上の実在を `curl /api/models` で確認済み。

## 共通制約

| 項目 | 値 |
|---|---|
| 実行 GPU | NVIDIA RTX 5090 (32GB VRAM, sm_120 / Blackwell) |
| 推論サーバ | vLLM 0.20.x (`scripts/serve_local_llm.sh` / `scripts/run_model_benchmark.sh`) |
| Attention backend | `FLASH_ATTN` (FlashInfer は Blackwell 未対応で除外) |
| KV cache dtype | fp8 (容量半減) |
| 出力形式 | OpenAI 互換 `response_format=json_schema` で xgrammar 既定の JSON 強制 |
| 必要要件 | 商用利用可・OSS・vLLM 対応 |

## 候補モデル

### 1. Qwen3.6-35B-A3B (unsloth NVFP4) — 第一採用候補

| 項目 | 値 |
|---|---|
| HF ID | `unsloth/Qwen3.6-35B-A3B-NVFP4` |
| 公開時期 | 2026 初頭 |
| パラメータ数 | 35B 総 / 3B アクティブ (MoE) |
| コンテキスト長 | 128K |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | NVFP4 (4bit) で 23GB on disk、GPU 上もほぼ同等 |
| 量子化フラグ | `--quantization compressed-tensors --enforce-eager` |
| 量子化形式 | unsloth が NVIDIA ModelOpt の NVFP4 で量子化し `compressed-tensors` フォーマットで配布 |
| 採用根拠 | (a) 最新世代 Qwen3.6、(b) MoE で 3B 並みの推論コストで 35B 知識量、(c) Blackwell ネイティブ FP4、(d) Apache 2.0、(e) 32GB に余裕で乗り KV cache も確保可能 |
| 注意 | 本家 Qwen は `qwen35moe` GGUF / BnB を vLLM 0.20.2 が未対応のため、unsloth NVFP4 が現状唯一の動作 ID |

### 2. Qwen2.5-32B-Instruct-AWQ — 控え候補

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| パラメータ数 | 32B (dense, INT4 量子化済み) |
| コンテキスト長 | 32K |
| ライセンス | Apache 2.0 (Qwen) |
| 32GB に収まる方法 | AWQ Marlin で約 18GB、KV cache fp8 込みで余裕 |
| 量子化フラグ | `--quantization awq_marlin --dtype auto --enforce-eager` |
| 採用根拠 | 公式 AWQ + Marlin カーネルでレイテンシ最速級。Qwen3.6 系の MoE が不安定な場合の確実な dense backup |

### 3. Qwen3.6-27B — 同世代 dense 比較

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen3.6-27B` |
| パラメータ数 | 27B (dense) |
| コンテキスト長 | 128K |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | bf16 で 54GB のため不可。**BnB NF4 オンザフライ**で約 14GB |
| 量子化フラグ | `--quantization bitsandbytes --dtype auto --enforce-eager` |
| 採用根拠 | Qwen3.6 世代の dense 比較。35B-A3B (MoE) と推論経路を分けて挙動を見る |
| 注意 | 公式 AWQ/GPTQ チェックポイントが現状未公開のため BnB を使う。BnB は AWQ Marlin / NVFP4 に比べてレイテンシ劣勢 |

### 4. Qwen3-14B — 世代 / サイズ ダウン比較

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen3-14B` |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 128K |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | bf16 で 28GB のため KV cache 確保不可。**BnB NF4** で約 7GB |
| 量子化フラグ | `--quantization bitsandbytes --dtype auto` |
| 採用根拠 | Qwen3 世代 14B の参考。SFT/DPO で重さがネックの場合の控え control / draft 候補 |

### 5. Phi-4-14B — 別系統 14B (MIT)

| 項目 | 値 |
|---|---|
| HF ID | `microsoft/phi-4` |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 16K |
| ライセンス | MIT |
| 32GB に収まる方法 | bf16 で 28GB のため OOM。**BnB NF4** で約 7GB |
| 量子化フラグ | `--quantization bitsandbytes --dtype auto --enforce-eager` |
| 採用根拠 | Qwen3-14B との同サイズ別系統 (MS / 合成データ重視訓練)。MIT で派生物制約なし |

### 6. Swallow-Llama-3.1-8B-Instruct — 日本語特化 8B 対照

| 項目 | 値 |
|---|---|
| HF ID | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3` |
| パラメータ数 | 8B (dense, Llama-3.1 + 日本語継続事前学習) |
| コンテキスト長 | 32K |
| ライセンス | Llama 3.1 Community License + Swallow License (商用可) |
| 32GB に収まる方法 | bf16 で 16GB、余裕あり |
| 量子化フラグ | `--dtype bfloat16` |
| 採用根拠 | 日本語特化 / サイズ最小のベースライン。Qwen 系が日本語精度で勝ち切れるかを定量化するための対照群 |

## 落選候補と理由

- **`Qwen/Qwen3.6-35B-A3B` (素モデル)**: bf16 は 70GB、AWQ/GPTQ 公式なし、BnB / GGUF は vLLM 0.20.2 で `qwen35moe` 未対応のため動作不可。NVFP4 (unsloth 提供) のみ動作確認できた
- **`Qwen/Qwen3.6-27B-Instruct(-AWQ)` / `Qwen3.6-14B`**: HF 未公開 (Qwen3 系は `-Instruct` を付けない統合モデル方式)
- **`meta-llama/Llama-3.3-8B-Instruct`**: Llama 3.3 は 70B のみ公開。Llama 3.1 + Swallow に統合
- **Gemma 2/3 27B**: ライセンスが再蒸留に制約
- **Plamo-100B**: 32GB に量子化でも乗らない

## 評価軸（仕様 #18 より再掲）

| 軸 | 計測方法 | 重み |
|---|---|---|
| 日本語評価精度 | `make eval` で Spearman / pairwise accuracy | **最重要** |
| JSON 出力安定性 | `response_format=json_schema` 強制時の成功率・軸別 SD (N=5) | 高 |
| 推論速度 | `scripts/measure_latency.py` の p50 / p95 | 中 |
| メモリ占有 | bf16 / 4bit 量子化での実 GPU 使用量 | 中 |
| 学習しやすさ | QLoRA + transformers/peft + bitsandbytes の参考実装数 | 中 |
| ライセンス | 商用可・再配布可・蒸留可 | 中 |

## 実測手順

SSH 先 (5090) で:

```bash
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score
cd backend && uv venv --python 3.13 --clear && cd ..
source backend/.venv/bin/activate
cd backend && uv sync && cd ..
uv pip install vllm bitsandbytes
uv pip uninstall flashinfer-python   # Blackwell では未対応で落ちる

bash scripts/run_model_benchmark.sh --all 2>&1 | tee /tmp/issue18_bench.log

python scripts/aggregate_benchmark_results.py --out reports/model_benchmarks/_summary.md
```

詳細は `docs/model_selection_v1.md` の「実測手順」セクション参照。

## 関連

- Issue #18 (本 Issue)
- Issue #5: eval ハーネス
- Issue #12 / `docs/inference_server_selection.md`: vLLM 採用根拠
- Issue #11 / #13 / #14 / #17: 採用モデルを参照する後続実装

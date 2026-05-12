# 判断モデル候補リスト (Issue #18)

会議貢献度スコアリングの判断 LLM 候補。**Mac で開発・SSH 越し RTX 5090 (32GB VRAM) で実行** を前提に絞り込んだ 5 モデル（HuggingFace 上の実在を `curl /api/models` で確認済み）。

## 共通制約

| 項目 | 値 |
|---|---|
| 実行 GPU | NVIDIA RTX 5090 (32GB VRAM, sm_120) |
| 推論サーバ | vLLM 0.20.x (`scripts/serve_local_llm.sh` / `scripts/run_model_benchmark.sh`) |
| 量子化 | bf16 を第一、収まらないものは AWQ / BitsAndBytes NF4 |
| 出力形式 | OpenAI 互換 `response_format=json_schema` で xgrammar 既定の JSON 強制 |
| 必要要件 | 商用利用可・OSS・vLLM 対応・日本語 instruct 動作可 |

## 候補モデル

### 1. Qwen3.6-27B (Alibaba) — 第一採用候補

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen3.6-27B` |
| 公開時期 | 2026 初頭 |
| パラメータ数 | 27B (dense) |
| コンテキスト長 | 128K (RoPE) |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | **BitsAndBytes NF4 でオンザフライ 4bit 量子化必須**（bf16 で 54GB のため不可。BnB NF4 で約 14GB） |
| 量子化フラグ | `--quantization bitsandbytes --dtype auto` |
| 得意領域 | 多言語（日本語含む）、推論、長文要約 |
| 採用根拠 | ユーザー指定 + 同世代 Qwen3 系が JMT-Bench / Nejumi で上位、Apache 2.0 で再蒸留制約なし |
| 注意 | 公式 AWQ/GPTQ チェックポイントが現状未公開のため、初回ロードに +5〜10 分かかる。`Instruct` 版も別 ID では未公開（Qwen3 系の命名規則は `-Instruct` を付けない統合モデル方式） |

### 2. Qwen3-14B (Alibaba) — 控え候補

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen3-14B` |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 128K |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | bf16 で 28GB（KV cache 用に MAX_MODEL_LEN を 8K 程度に絞る） |
| 量子化フラグ | `--dtype bfloat16` |
| 採用根拠 | Qwen3.6-27B が SFT/DPO で重い場合の controller / draft 候補。同系統で tokenizer 共有 |

### 3. Qwen2.5-32B-Instruct-AWQ (Alibaba) — 公式 AWQ ありの 32B 比較対象

| 項目 | 値 |
|---|---|
| HF ID | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| パラメータ数 | 32B (dense, INT4 量子化済み) |
| コンテキスト長 | 32K |
| ライセンス | Apache 2.0 (Qwen) |
| 32GB に収まる方法 | AWQ Marlin で約 18GB、bf16 KV cache 込みで余裕あり |
| 量子化フラグ | `--quantization awq_marlin --dtype auto` |
| 採用根拠 | Qwen3.6-27B (Qwen3 世代) と Qwen2.5 世代の **世代間比較**。AWQ 公式配布があるため初回ロードが速い |

### 4. Swallow-Llama-3.1-8B-Instruct-v0.3 (東京工業大) — 日本語特化対照群

| 項目 | 値 |
|---|---|
| HF ID | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3` |
| パラメータ数 | 8B (dense, Llama-3.1 ベース + 日本語継続事前学習) |
| コンテキスト長 | 32K |
| ライセンス | Llama 3.1 Community License + Swallow License (商用可) |
| 32GB に収まる方法 | bf16 で 16GB、余裕あり |
| 量子化フラグ | `--dtype bfloat16` |
| 採用根拠 | **サイズ最小 + 日本語特化** の対照群。27B/32B クラスが日本語精度で勝てない場合の代替ライン |

### 5. Phi-4-14B (Microsoft) — 別系統 14B

| 項目 | 値 |
|---|---|
| HF ID | `microsoft/phi-4` |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 16K |
| ライセンス | MIT |
| 32GB に収まる方法 | bf16 で 28GB |
| 量子化フラグ | `--dtype bfloat16` |
| 採用根拠 | Qwen3-14B との **同サイズ・別系統** 比較。MIT で再配布／蒸留制約なし |
| 注意 | 合成データ重視訓練・指示追従が強いが、日本語ベンチでは Qwen / Swallow に劣ることが多い |

## 落選候補と理由

- **Qwen/Qwen3.6-27B-Instruct(-AWQ/-GPTQ)**: HF 上に未公開 (401)。本家が公式量子化版を配布次第、Qwen3.6-27B + BnB を置き換える
- **Qwen/Qwen3.6-14B 系**: HF 上に未公開 (401)
- **meta-llama/Llama-3.3-8B-Instruct**: Llama 3.3 は 70B のみ公開で 8B は無い (401)。Llama 3.1 + Swallow に統合
- **Qwen/Qwen2.5-14B-Instruct**: Qwen3-14B と冗長
- **Gemma 2/3 27B**: ライセンスが再蒸留・派生物に制約あり
- **Plamo-100B**: サイズ超過 (32GB に量子化でも乗らない)
- **DeepSeek-V2-Lite 16B (MoE)**: PEFT 実装の成熟度が SFT/DPO 用途で不足

## 評価軸（仕様 #18 より再掲）

| 軸 | 計測方法 | 重み |
|---|---|---|
| 日本語評価精度 | `make eval` で Spearman / pairwise accuracy | **最重要** |
| JSON 出力安定性 | `response_format=json_schema` 強制時の成功率・軸別 SD (N=5) | 高 |
| 推論速度 | `scripts/measure_latency.py` の p50 / p95 | 中 |
| メモリ占有 | bf16 / 4bit 量子化での実 GPU 使用量 | 中 |
| 学習しやすさ | QLoRA + transformers/peft + bitsandbytes の参考実装数 | 中 |
| ライセンス | 商用可・再配布可・蒸留可 | 中 |

## 実行手順

```bash
# SSH 先 (5090) で
cd ~/mtg-score/meeting-score
source backend/.venv/bin/activate
bash scripts/run_model_benchmark.sh --all 2>&1 | tee /tmp/issue18_bench.log

# 集計
python scripts/aggregate_benchmark_results.py --out reports/model_benchmarks/_summary.md
```

結果 JSON は `reports/model_benchmarks/{served_name}/{ts}_*.json`。`docs/model_selection_v1.md` の表に転記する。

## 関連

- Issue #18 (本 Issue)
- Issue #5: eval ハーネス（Spearman / pairwise / Top-K / 安定性）
- Issue #12 / `docs/inference_server_selection.md`: vLLM を採用済
- Issue #11 / #13 / #14 / #16: 採用モデルを参照する後続実装

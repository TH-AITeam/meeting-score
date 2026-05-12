# 判断モデル候補リスト (Issue #18)

会議貢献度スコアリングの判断 LLM 候補。**Mac で開発・SSH 越し RTX 5090 (32GB VRAM) で実行** を前提に絞り込んだ 5 モデル。

## 共通制約

| 項目 | 値 |
|---|---|
| 実行 GPU | NVIDIA RTX 5090 (32GB VRAM, sm_120) |
| 推論サーバ | vLLM (`scripts/serve_local_llm.sh`) |
| 量子化 | bf16 を第一、収まらないものは AWQ INT4 (vLLM が公式対応) |
| 出力形式 | OpenAI 互換 `response_format=json_schema` で JSON Schema 強制 |
| 必要要件 | 商用利用可・OSS・vLLM 対応・日本語 instruct 学習済み |

## 候補モデル

### 1. Qwen3.6-27B-Instruct (Alibaba)

| 項目 | 値 |
|---|---|
| 公開時期 | 2026 初頭 |
| パラメータ数 | 27B (dense) |
| コンテキスト長 | 128K (RoPE) |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | **AWQ INT4 量子化必須**（bf16 では 54GB で不可。AWQ で約 14GB） |
| 得意領域 | 多言語（日本語含む）、推論、長文要約 |
| 採用根拠 | 同シリーズの 14B/30B-MoE が Nejumi/JMT-Bench で上位定着。サイズ・JP・ライセンス・vLLM 公式 AWQ 配布の四拍子 |
| 注意 | sm_120 (Blackwell) で動かすには vLLM 0.6+ / FlashAttn 3 ビルドが必要 |

### 2. Qwen3-14B-Instruct (Alibaba)

| 項目 | 値 |
|---|---|
| 公開時期 | 2025 後半 |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 128K |
| ライセンス | Apache 2.0 |
| 32GB に収まる方法 | bf16 で 28GB（GPU メモリ余裕は少ない、コンテキスト 32K 程度に制限）／AWQ INT4 で 8GB |
| 得意領域 | 多言語、コード、推論。Qwen3.6-27B のサイズダウン版として速度／コストの基準 |
| 採用根拠 | 27B が SFT/DPO で重くなる場合の controller / draft 候補にも使える |

### 3. Llama-3.3-8B-Instruct (Meta)

| 項目 | 値 |
|---|---|
| 公開時期 | 2024 末 |
| パラメータ数 | 8B (dense) |
| コンテキスト長 | 128K |
| ライセンス | Llama 3 Community License（条件付き商用可、月間 7 億 MAU 制限） |
| 32GB に収まる方法 | bf16 で 16GB、AWQ で 4GB |
| 得意領域 | 英語強い、汎用 instruct |
| 採用根拠 | サイズ最小・推論最速の **下限ライン** として比較表に置く。日本語ベンチで他に劣るかを定量化 |
| 注意 | Llama 3 Community License は厳密には Apache 2.0/MIT より制約が強い |

### 4. Swallow-Llama-3.1-8B-Instruct (東京工業大)

| 項目 | 値 |
|---|---|
| 公開時期 | 2024 |
| パラメータ数 | 8B (dense, Llama-3.1 ベースの日本語継続事前学習) |
| コンテキスト長 | 32K |
| ライセンス | Llama 3.1 Community License + Swallow License (商用可) |
| 32GB に収まる方法 | bf16 で 16GB |
| 得意領域 | **日本語**（タスク Specific / instruct）、汎用日本語 |
| 採用根拠 | 同サイズの英語ベース Llama-3.3-8B との **日本語性能差**を取るための対照群 |

### 5. Phi-4-14B (Microsoft)

| 項目 | 値 |
|---|---|
| 公開時期 | 2024 末 |
| パラメータ数 | 14B (dense) |
| コンテキスト長 | 16K |
| ライセンス | MIT |
| 32GB に収まる方法 | bf16 で 28GB |
| 得意領域 | 推論・指示追従（合成データ重視訓練） |
| 採用根拠 | Qwen3-14B との **同サイズ・別系統** の比較。MIT で再配布／蒸留制約なし |
| 注意 | 日本語向けの追加学習は弱め。Swallow に対する逆ベンチとして機能 |

## 落選候補と理由

- **Gemma 2 27B / Gemma 3 27B**: ライセンス（Gemma Terms of Use）が再蒸留に制約あり → SFT/DPO のしやすさで落選
- **Plamo-100B-Pretrained**: 32GB では 4-bit でも動かない（推定 50GB+）
- **Mistral Small 3 24B**: vLLM 公式 AWQ がリリース初期で不安定 → 候補から外す（再評価候補に残す）
- **DeepSeek-V2-Lite 16B (MoE)**: 32GB に AWQ で収まるが、MoE は SFT/DPO 用 PEFT 実装が成熟しきっておらず Issue #13/#14 のリスク

## 評価軸（再掲: 仕様より）

| 軸 | 計測方法 | 重み |
|---|---|---|
| 日本語評価精度 | サンプル会議3本 + ゴールデンで Spearman / pairwise accuracy（`make eval`） | **最重要** |
| JSON 出力安定性 | 100 発言で `response_format=json_schema` 強制時の成功率と軸別 SD | 高 |
| 推論速度 | 1 発言あたり p50 / p95 レイテンシ（`scripts/run_model_benchmark.sh`） | 中 |
| メモリ占有 | bf16 / AWQ INT4 での実 GPU 使用量（`nvidia-smi`） | 中 |
| 学習しやすさ | QLoRA 設定 (transformers + peft + bitsandbytes) の参考実装数 | 中 |
| ライセンス | 商用可・再配布可・蒸留可 | 中 |

実機計測は `scripts/run_model_benchmark.sh` を SSH 先 (RTX 5090) で 1 モデルずつ回し、結果 JSON を `reports/model_benchmarks/{model}/{ts}.json` に出力する。集計は `docs/model_selection_v1.md` の表に転記する。

## 関連

- Issue #18 (本 Issue)
- Issue #5: eval ハーネス（Spearman / pairwise / Top-K / 安定性）
- Issue #12 / `docs/inference_server_selection.md`: vLLM を採用済
- Issue #11 / #13 / #14 / #16: 採用モデルを参照する後続実装

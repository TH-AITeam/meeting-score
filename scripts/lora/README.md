# Qwen3.5-9B bf16 LoRA 学習

国会会議録の蒸留データ（`data/annotations/kokkai/distill/train.jsonl` / `val.jsonl`、messages 形式）で
発言評価モデルを LoRA 微調整する。フレームワークは **Unsloth**、量子化はせず
**ベース重みを bf16 のまま読み込む 16bit LoRA**（Qwen3.5 系は 4bit QLoRA だと
量子化差分が大きく非推奨、という Unsloth の指摘に従う）。

> 本ディレクトリは **学習スクリプトのみ**。`train.jsonl` / `val.jsonl` の生成
> （生会議録 → 教師ラベル蒸留）は `data/annotations/kokkai/distill/README.md` を参照。

## セットアップ（backend とは別 venv 推奨）

RTX 5090 (Blackwell) は CUDA 12.8+ ビルドの PyTorch が必要。

```bash
python -m venv scripts/lora/.venv-lora && source scripts/lora/.venv-lora/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r scripts/lora/requirements.txt
```

## 学習

```bash
# 既定: data/annotations/kokkai/distill/{train,val}.jsonl を読み、outputs/qwen35-9b-lora に出力
python scripts/lora/train_lora.py

# vLLM 配信用に 16bit マージ済みモデルも書き出す
python scripts/lora/train_lora.py --save-merged
```

主なオプション（`--help` で全件）:

| オプション | 既定 | 説明 |
|---|---|---|
| `--model-id` | `Qwen/Qwen3.5-9B` | ベースモデルの HF ID |
| `--train` / `--val` | `data/annotations/kokkai/distill/*.jsonl` | messages 形式 JSONL |
| `--max-seq-len` | `4096` | 最大系列長。OOM 時は下げる |
| `--epochs` | `2` | エポック数 |
| `--batch-size` / `--grad-accum` | `1` / `8` | 実効バッチ = 8 |
| `--lora-r` / `--lora-alpha` | `16` / `16` | LoRA ランク / alpha |
| `--lr` | `2e-4` | 学習率 |
| `--save-merged` | off | 16bit マージ済みモデルを保存 |

## 学習ログ（wandb）

学習中の loss / 学習率 / eval などは [wandb](https://wandb.ai) に記録される（既定で有効）。
API キーは **`backend/.env` の `WANDB_API_KEY`** から読み込む（`OPENAI_API_KEY` 等と同じ場所）。

```bash
# backend/.env に追記
echo 'WANDB_API_KEY=<あなたのキー>' >> backend/.env
```

| オプション | 既定 | 説明 |
|---|---|---|
| `--wandb-project` | `meeting-score-lora` | wandb プロジェクト名 |
| `--wandb-run-name` | (自動) | run 名 |
| `--no-wandb` | off | wandb ロギングを無効化 |

`WANDB_API_KEY` が無い／`wandb` 未インストールの場合は、警告を出してログ無効で学習を続行する。

## RTX 5090 (32GB) のメモリ目安

- bf16 9B はベース重みだけで約 18GB。`per_device_batch_size=1` +
  grad checkpointing（既定）+ `adamw_8bit` 前提で `max-seq-len 4096` 程度が安全圏。
- **OOM したら**: `--max-seq-len 2048`、または `--grad-accum` を上げて
  実効バッチを保ちつつ `--batch-size 1` を維持する。

## 学習プロンプト ＝ 本番プロンプトの一致（重要）

`train.jsonl` の `user` フィールドは、本番推論の
`backend/app/evaluators/prompt.py::build_prompt()` の出力そのものであること。
学習時は Qwen チャットテンプレートを適用し、損失は **assistant 応答部のみ**
（`train_on_responses_only`）に限定する。thinking は既定で無効化して
本番（温度 0・JSON Schema 強制）と同じく直接 JSON を出させる。

## 学習後の配信

`outputs/qwen35-9b-lora/merged_16bit`（`--save-merged` で生成）を vLLM で配信:

```bash
MODEL="$(pwd)/outputs/qwen35-9b-lora/merged_16bit" bash scripts/serve_local_llm.sh
```

`backend/app/evaluators/local_evaluator.py`（OpenAI 互換クライアント）から
この endpoint を指して評価に使う。

## DPO 学習（Issue #15）

SFT 済み LoRA に対し、ペアワイズ選好で **順位の納得感** を直接最適化する
（絶対スコア精度ではなく「どちらの発言が会議を前進させたか」を学ぶ）。

> 配置メモ: Issue #15 は `training/` 配下を想定していたが、既存の学習基盤が
> `scripts/lora/`（train_lora / eval_lora、専用 venv）に集約されているため、
> DPO もここに置いて規約・venv を共有する。

### 1. DPO データを作る（`scripts/build_dpo_dataset.py`）

`{prompt, chosen, rejected, meta}` 形式（winner=chosen / loser=rejected）。
`prompt`/`chosen`/`rejected` は本番推論と同一プロンプト・本番スキーマ準拠 JSON。

```bash
# gold/合成ペア（#5/#7）から
python scripts/build_dpo_dataset.py --pairs data/annotations/gold/v1/pairs.jsonl

# gold 未整備時のブートストラップ: 既存 distill ラベルのスコア差からペアを合成
python scripts/build_dpo_dataset.py --synthesize-from-labels
# -> data/dpo/v1/{train,val}.jsonl
```

> 本来は各ペアを **SFT モデルで再評価** して chosen/rejected JSON を作る（#15）。
> 現状は GPU 無しでも回るよう既存ラベルを評価 JSON として流用する。SFT モデルでの
> 再評価は vLLM 配信 + `--eval-source` 拡張の将来作業。

### 2. DPO を回す（`scripts/lora/dpo_train.py`）

設定は `scripts/lora/configs/dpo_v1.yaml`（β=0.1 / lr=5e-7 / epochs=1）。
**先に SFT（train_lora.py）を済ませて `outputs/qwen35-9b-lora` を用意**しておくこと。

```bash
source scripts/lora/.venv-lora/bin/activate
python scripts/lora/dpo_train.py                 # configs/dpo_v1.yaml 既定
python scripts/lora/dpo_train.py --beta 0.05     # β を振って eval
python scripts/lora/dpo_train.py --save-merged   # vLLM 配信用 16bit マージ
# -> outputs/qwen35-9b-dpo
```

β は 0.05〜0.2 で振る（大きすぎると SFT 性能を破壊）。trl の `DPOTrainer` を
Unsloth の `PatchDPOTrainer` で最適化し、参照モデルはアダプタ無効化で代用（省メモリ）。

### 3. SFT 単独 vs SFT+DPO を比較（`scripts/lora/eval_pairwise.py`）

完了条件の指標（ペアワイズ accuracy / Top-5 Jaccard）を測る。各アダプタで実行して差分を見る。

```bash
python scripts/lora/eval_pairwise.py --adapter outputs/qwen35-9b-lora   # SFT
python scripts/lora/eval_pairwise.py --adapter outputs/qwen35-9b-dpo    # SFT+DPO
```

完了条件: ペアワイズ accuracy が SFT-only より **+3pt 以上** / Top-5 Jaccard 改善 /
人手スポットチェックで「Top に来る発言の納得感が上がった」。これらは
**gold ペア（#5）+ 学習済み SFT モデル（#14）+ GPU** が揃って初めて測れる。

# #13 [training] SFT 学習スクリプト(QLoRA)

**Labels**: `training`, `P0`
**Milestone**: v0.6

## 前提

**先に #17 で判断モデルを選定すること**。 ここでは「採用ベースモデル」と抽象的に書く。

## 概要

#12 のデータセットで、採用ベースモデル(#17)を QLoRA で SFT する。Kaggle Jigsaw でやった構成の流用でOK。

## 技術選定

- **学習フレームワーク**: `trl` の `SFTTrainer` + `peft` (LoRA)
  - 速度優先なら `unsloth` も検討(対応モデルなら 2倍速くらい)
- **量子化**: bitsandbytes 4-bit (NF4)、LoRA は r=16, α=32 から開始
- **prompt format**: 採用ベースモデルの chat template に合わせる

## ディレクトリ

```
training/
  __init__.py
  configs/
    sft_v1.yaml              # 採用モデル名は config 内で指定
  sft_train.py               # メインスクリプト
  callbacks.py               # eval_callback (#4 メトリクスを学習中に回す)
  merge_lora.py              # LoRA を base にマージして1モデルにする
checkpoints/
  meeting-sft-v1/
```

## やること

- [ ] `training/configs/sft_v1.yaml`
  - base_model: 採用モデル名(#17)
  - lr=2e-4, epochs=2〜3, max_len=4096, batch=4, grad_accum=4
  - target_modules: モデルの構造に合わせて指定(典型は `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
- [ ] `sft_train.py` 実装
- [ ] WandB or TensorBoard で loss / eval ロギング
- [ ] `EvalCallback`: 100step ごとに #4 の eval ハーネスを `val.jsonl` で回す
- [ ] `merge_lora.py` で LoRA をマージ
- [ ] マージ後モデルを推論サーバ(#11)で立ち上げ、サンプル会議3本で eval を回す
- [ ] 結果を `docs/training_log_v1.md` にまとめる

## 完了条件

- val ペアワイズ accuracy が pre-train ベースモデルより +5pt 以上
- val Spearman が +0.05 以上
- 学習後モデルが推論サーバで動き、#11 の `LocalEvaluator` でそのまま使える

## ハードウェア要件

- A100 80GB 1枚で 7B クラス QLoRA: 5000件 / 3 epoch / 4096 tokens で **3〜6h** 想定
- 採用モデルのパラメータ数で前後する。14B 級なら QLoRA でも 80GB ギリ、grad_accum を増やす

## 注意

- `train_on_completions_only` を有効にして、ユーザー入力部分のロスを除外
- LoRA を base に**マージしない**生 adapter のままだと推論サーバ側で adapter ロード対応が必要。最初はマージ推奨
- 採用ベースモデルが Instruct チューニング済みなら system プロンプトの扱いに注意(モデルごとに違う)

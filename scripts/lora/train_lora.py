#!/usr/bin/env python3
"""Qwen3.5-9B を bf16 LoRA で微調整する学習スクリプト (Unsloth)。

国会会議録の蒸留データ（`data/annotations/kokkai/distill/train.jsonl` / `val.jsonl`、
messages 形式）を読み込み、発言評価モデルを LoRA で蒸留する。

なぜ bf16 LoRA か:
    Unsloth のドキュメントによれば Qwen3.5 系は 4bit QLoRA だと量子化差分が
    通常より大きく非推奨。本スクリプトはベース重みを bf16 のまま読み込み
    (load_in_4bit=False) LoRA を載せる「16bit LoRA」構成にしている。

データ形式 (1 行 1 サンプル):
    {"messages": [
        {"role": "user", "content": "<本番 build_prompt() の出力そのまま>"},
        {"role": "assistant", "content": "{\\"speech_type\\":...,\\"scores\\":{...},...}"}
    ]}

    `user` は本番推論プロンプト（backend/prompts/utterance_eval.txt 由来）、
    `assistant` は本番スキーマ準拠の JSON 文字列であること
    （data/annotations/kokkai/distill/README.md の「train と inference を一致させる」原則）。

使い方:
    # 既定（train/val を自動探索、LoRA アダプタを出力）
    python scripts/lora/train_lora.py

    # データ・出力・主要ハイパラを上書き
    python scripts/lora/train_lora.py \
        --train data/annotations/kokkai/distill/train.jsonl \
        --val   data/annotations/kokkai/distill/val.jsonl \
        --output-dir outputs/qwen35-9b-lora \
        --epochs 2 --max-seq-len 4096 --lora-r 16

    # vLLM 配信用に 16bit マージ済みモデルも書き出す
    python scripts/lora/train_lora.py --save-merged

RTX 5090 (Blackwell, 32GB) での目安:
    bf16 9B はベース重みだけで ~18GB。grad checkpointing 前提で
    per_device_batch_size=1, max_seq_len=4096 程度が安全圏。
    OOM する場合は --max-seq-len 2048 か --grad-accum を上げて調整。
"""

from __future__ import annotations

# Unsloth は torch / transformers より先に import するとパッチが最適に効く。
import unsloth  # noqa: F401  # isort:skip

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_lora")

REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILL_DIR = REPO_ROOT / "data" / "annotations" / "kokkai" / "distill"
# wandb の API キーは backend/.env に置く（OPENAI_API_KEY 等と同じ場所）。
ENV_PATH = REPO_ROOT / "backend" / ".env"

# Qwen のチャットテンプレートにおける指示部・応答部のマーカー。
# train_on_responses_only がこれ以降のトークンだけを損失計算の対象にする。
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3.5-9B bf16 LoRA fine-tuning (Unsloth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # モデル / データ
    p.add_argument("--model-id", default="Qwen/Qwen3.5-9B", help="ベースモデルの HF ID")
    p.add_argument(
        "--train",
        default=str(DISTILL_DIR / "train.jsonl"),
        help="学習データ (messages 形式 JSONL)",
    )
    p.add_argument(
        "--val",
        default=str(DISTILL_DIR / "val.jsonl"),
        help="検証データ。存在しなければ eval をスキップ",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "qwen35-9b-lora"),
        help="LoRA アダプタ・チェックポイントの出力先",
    )
    # LoRA ハイパーパラメータ
    p.add_argument("--lora-r", type=int, default=16, help="LoRA ランク")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--max-seq-len", type=int, default=4096, help="最大系列長")
    # 学習スケジュール
    p.add_argument("--epochs", type=float, default=2.0, help="エポック数")
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="ステップ数で打ち切る場合に指定 (>0 で epochs より優先)",
    )
    p.add_argument("--batch-size", type=int, default=1, help="GPU あたりバッチサイズ")
    p.add_argument(
        "--grad-accum", type=int, default=8, help="勾配累積ステップ (実効バッチ = batch*accum)"
    )
    p.add_argument("--lr", type=float, default=2e-4, help="学習率")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="ウォームアップ比率")
    p.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    p.add_argument("--seed", type=int, default=3407, help="乱数シード")
    # ロギング / 保存
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Qwen3 系の thinking を無効化してテンプレート展開 (既定: 無効)",
    )
    p.set_defaults(enable_thinking=False)
    p.add_argument(
        "--save-merged",
        action="store_true",
        help="学習後に 16bit マージ済みモデルも保存 (vLLM 配信用)",
    )
    # wandb（学習ログ）
    p.add_argument(
        "--wandb-project",
        default="meeting-score-lora",
        help="wandb のプロジェクト名",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="wandb の run 名（未指定なら wandb が自動命名）",
    )
    p.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="wandb ロギングを無効化（既定: 有効）",
    )
    p.set_defaults(use_wandb=True)
    return p.parse_args()


def _load_dotenv(path: Path) -> None:
    """backend/.env を環境変数に読み込む（python-dotenv が無くても動く簡易版）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_wandb(args: argparse.Namespace) -> str:
    """backend/.env の WANDB_API_KEY で wandb にログインし、report_to を返す。

    キーが無い・wandb 未インストールの場合はログのみ無効化して学習は続行する。
    """
    if not args.use_wandb:
        logger.info("wandb は --no-wandb で無効化されています")
        return "none"

    _load_dotenv(ENV_PATH)
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.warning(
            "WANDB_API_KEY が見つかりません（%s に WANDB_API_KEY=... を追加してください）。"
            "wandb ロギングを無効化して続行します。",
            ENV_PATH,
        )
        return "none"

    try:
        import wandb
    except ImportError:
        logger.warning("wandb 未インストール（pip install wandb）。ロギングを無効化して続行します。")
        return "none"

    wandb.login(key=api_key)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    logger.info("wandb 有効: project=%s run=%s", args.wandb_project, args.wandb_run_name or "(自動)")
    return "wandb"


def load_model(args: argparse.Namespace):
    """bf16 でベースモデルを読み込み、LoRA アダプタを付与する。"""
    logger.info("ベースモデルを bf16 で読み込み: %s", args.model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,  # bf16 固定（4bit QLoRA は使わない）
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # "unsloth" は通常の grad checkpointing より VRAM を節約する。
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def build_dataset(args: argparse.Namespace, tokenizer):
    """messages 形式 JSONL を読み、chat template を適用して text 列を作る。"""

    def format_messages(batch):
        texts = []
        for messages in batch["messages"]:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=args.enable_thinking,
                )
            except TypeError:
                # enable_thinking を受け付けないテンプレート向けフォールバック。
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            texts.append(text)
        return {"text": texts}

    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(f"学習データが見つからない: {train_path}")
    logger.info("学習データ読み込み: %s", train_path)
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    train_ds = train_ds.map(format_messages, batched=True, remove_columns=train_ds.column_names)

    val_ds = None
    val_path = Path(args.val)
    if val_path.exists():
        logger.info("検証データ読み込み: %s", val_path)
        val_ds = load_dataset("json", data_files=str(val_path), split="train")
        val_ds = val_ds.map(
            format_messages, batched=True, remove_columns=val_ds.column_names
        )
    else:
        logger.warning("検証データが無いため eval をスキップ: %s", val_path)

    logger.info("train=%d 件, val=%s 件", len(train_ds), len(val_ds) if val_ds else "なし")
    return train_ds, val_ds


def main() -> None:
    args = parse_args()
    report_to = setup_wandb(args)
    model, tokenizer = load_model(args)
    train_ds, val_ds = build_dataset(args, tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        # bf16 学習（RTX 5090 / Blackwell は bf16 ネイティブ対応）
        bf16=True,
        fp16=False,
        # 評価も bf16 のまま実行する。これをしないと eval 時に巨大語彙×長系列の
        # logits を fp32 変換（tensor.float()）して 15GB 級のスパイクが発生し OOM する。
        bf16_full_eval=True,
        # eval で全 logits を蓄積せず loss のみ計算（メモリスパイク抑制）。
        prediction_loss_only=True,
        per_device_eval_batch_size=1,
        optim="adamw_8bit",  # オプティマイザ状態を 8bit 化して VRAM 節約
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps if val_ds is not None else None,
        seed=args.seed,
        report_to=report_to,
        run_name=args.wandb_run_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    # 損失を assistant 応答部のみに限定（プロンプト部は学習しない）。
    trainer = train_on_responses_only(
        trainer,
        instruction_part=QWEN_INSTRUCTION_PART,
        response_part=QWEN_RESPONSE_PART,
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.max_memory_reserved() / 1e9
        logger.info("GPU: %s (%.1f GB), 起動時予約: %.1f GB", gpu.name, gpu.total_memory / 1e9, reserved)

    logger.info("学習開始")
    trainer.train()

    out = Path(args.output_dir)
    logger.info("LoRA アダプタを保存: %s", out)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    if args.save_merged:
        merged_dir = out / "merged_16bit"
        logger.info("16bit マージ済みモデルを保存 (vLLM 配信用): %s", merged_dir)
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit"
        )

    logger.info("完了")


if __name__ == "__main__":
    main()

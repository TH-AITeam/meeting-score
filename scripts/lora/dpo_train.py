#!/usr/bin/env python3
"""SFT 済み LoRA に対して DPO を回す学習スクリプト (Issue #15, Unsloth + trl)。

`scripts/lora/train_lora.py`（SFT）の出力アダプタから初期化し、
`scripts/build_dpo_dataset.py` が出した ``{prompt, chosen, rejected}`` 形式の
ペアワイズ選好で「順位の納得感」を直接最適化する。

設定は `scripts/lora/configs/dpo_v1.yaml`（β=0.1, lr=5e-7, epochs=1）を既定で読み、
CLI 引数で個別に上書きできる。学習プロンプトは本番推論と一致させる（SFT と同様）。

使い方:
    source scripts/lora/.venv-lora/bin/activate
    # まず SFT を済ませて outputs/qwen35-9b-lora を用意しておくこと（train_lora.py）
    python scripts/lora/dpo_train.py                       # configs/dpo_v1.yaml 既定
    python scripts/lora/dpo_train.py --beta 0.05 --epochs 1
    python scripts/lora/dpo_train.py --save-merged         # vLLM 配信用 16bit マージ

RTX 5090 (Blackwell, 32GB) 目安:
    bf16 9B + 参照モデル分のメモリに注意。Unsloth は参照モデルを別途持たず
    アダプタ無効化で参照を作るため省メモリ。OOM 時は --max-seq-len を下げる。
"""

from __future__ import annotations

# Unsloth は torch / transformers / trl より先に import するとパッチが最適に効く。
import unsloth  # noqa: F401  # isort:skip

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

# trl.DPOTrainer を Unsloth 最適化でパッチする（DPOTrainer をインスタンス化する前に呼ぶ）。
PatchDPOTrainer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dpo_train")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dpo_v1.yaml"
ENV_PATH = REPO_ROOT / "backend" / ".env"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DPO fine-tuning from SFT LoRA (Unsloth + trl)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 設定ファイル")
    # 以下は指定時のみ YAML を上書き（既定 None）
    p.add_argument("--model-id", default=None)
    p.add_argument("--sft-adapter", default=None, help="初期化元の SFT LoRA アダプタ")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--epochs", type=float, default=None, dest="num_train_epochs")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--save-merged", action="store_true", help="16bit マージ済みも保存")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.set_defaults(use_wandb=True)
    p.add_argument("--wandb-project", default="meeting-score-lora")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """YAML を読み、CLI で指定された項目だけ上書きする。"""
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    overrides = {
        "model_id": args.model_id,
        "sft_adapter": args.sft_adapter,
        "output_dir": args.output_dir,
        "train": args.train,
        "val": args.val,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_seq_len": args.max_seq_len,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _resolve(path_str: str) -> str:
    """相対パスはリポジトリルート基準に解決する。"""
    p = Path(path_str)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def _load_dotenv(path: Path) -> None:
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
    """backend/.env の WANDB_API_KEY で wandb を有効化（無ければログ無効で続行）。"""
    if not args.use_wandb:
        return "none"
    _load_dotenv(ENV_PATH)
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY が無いため wandb を無効化して続行します（%s）", ENV_PATH
        )
        return "none"
    try:
        import wandb
    except ImportError:
        logger.warning("wandb 未インストール。ロギング無効で続行します。")
        return "none"
    wandb.login(key=os.environ["WANDB_API_KEY"])
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    return "wandb"


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    report_to = setup_wandb(args)

    sft_adapter = _resolve(cfg["sft_adapter"])
    if not Path(sft_adapter).exists():
        raise FileNotFoundError(
            f"SFT アダプタが見つかりません: {sft_adapter}\n"
            "先に scripts/lora/train_lora.py で SFT を実行してください。"
        )

    logger.info("SFT アダプタから DPO を初期化: %s", sft_adapter)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter,  # adapter_config が base を指す
        max_seq_length=cfg["max_seq_len"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    # DPO でも LoRA を学習対象として有効化（SFT と同じ構成）。
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    train_path = _resolve(cfg["train"])
    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds = None
    val_path = Path(_resolve(cfg["val"]))
    if val_path.exists():
        val_ds = load_dataset("json", data_files=str(val_path), split="train")
    logger.info(
        "DPO データ: train=%d, val=%s", len(train_ds), len(val_ds) if val_ds else "なし"
    )

    dpo_config = DPOConfig(
        output_dir=_resolve(cfg["output_dir"]),
        beta=cfg["beta"],
        loss_type=cfg.get("loss_type", "sigmoid"),
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type="cosine",
        max_length=cfg["max_seq_len"],
        max_prompt_length=cfg.get("max_prompt_len", cfg["max_seq_len"] // 2),
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_strategy="steps",
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=cfg["eval_steps"] if val_ds is not None else None,
        seed=cfg["seed"],
        report_to=report_to,
        run_name=args.wandb_run_name,
    )

    # ref_model=None: Unsloth はアダプタ無効化で参照を作り、別モデルを持たず省メモリ。
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB)", gpu.name, gpu.total_memory / 1e9)

    logger.info(
        "DPO 学習開始 (β=%s, lr=%s, epochs=%s)",
        cfg["beta"],
        cfg["learning_rate"],
        cfg["num_train_epochs"],
    )
    trainer.train()

    out = Path(_resolve(cfg["output_dir"]))
    logger.info("DPO 済み LoRA を保存: %s", out)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    if args.save_merged:
        merged = out / "merged_16bit"
        logger.info("16bit マージ済みを保存 (vLLM 配信用): %s", merged)
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
    logger.info("完了")


if __name__ == "__main__":
    main()

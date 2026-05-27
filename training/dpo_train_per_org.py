#!/usr/bin/env python3
"""組織別 DPO LoRA 学習 CLI (Issue #82).

通常のテスト環境では重い training 依存を import しない。実学習時だけ
``datasets`` / ``torch`` / ``transformers`` / ``peft`` / ``trl`` を読み込む。
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "dpo_org_v1.yaml"


def load_training_config(path: Path, org_id: str, llm_model: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("{{ org_id }}", org_id).replace("{{ llm_model }}", llm_model)
    return yaml.safe_load(text) or {}


def write_training_meta(
    output_dir: Path,
    *,
    org_id: str,
    base_model: str,
    dataset_path: Path,
    dataset_hash: str,
    init_adapter: str | None,
    eval_metrics: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "org_id": org_id,
        "base_model": base_model,
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "init_adapter": init_adapter,
        "eval_metrics": eval_metrics or {},
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DPO で組織別 LoRA アダプタを学習する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--org-id", required=True)
    p.add_argument("--train-jsonl", required=True, help="DPO JSONL (prompt/chosen/rejected)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--init-adapter", default=None, help="増分学習の初期アダプタ")
    p.add_argument("--dataset-hash", default="")
    p.add_argument("--mock", action="store_true", help="GPU 学習を行わずメタファイルだけ作る")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    cfg = load_training_config(Path(args.config), args.org_id, args.model_id)
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.train_jsonl)

    if args.mock:
        (output_dir / "adapter_config.json").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": args.model_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "adapter_model.safetensors").write_bytes(b"mock adapter\n")
        write_training_meta(
            output_dir,
            org_id=args.org_id,
            base_model=args.model_id,
            dataset_path=dataset_path,
            dataset_hash=args.dataset_hash,
            init_adapter=args.init_adapter,
        )
        return

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo = cfg.get("dpo", {})
    lora = cfg.get("lora", {})
    lr = dpo.get("incremental_learning_rate") if args.init_adapter else dpo.get("learning_rate")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora.get("r", 16),
                lora_alpha=lora.get("alpha", 16),
                lora_dropout=lora.get("dropout", 0.0),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

    train_ds = load_dataset("json", data_files=str(dataset_path), split="train")
    train_args = DPOConfig(
        output_dir=str(output_dir),
        beta=dpo.get("beta", 0.1),
        learning_rate=lr,
        num_train_epochs=dpo.get("epochs", 1),
        per_device_train_batch_size=dpo.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=dpo.get("gradient_accumulation_steps", 8),
        warmup_ratio=dpo.get("warmup_ratio", 0.03),
        weight_decay=dpo.get("weight_decay", 0.01),
        bf16=True,
        fp16=False,
        save_strategy="no",
        report_to="none",
    )
    trainer = DPOTrainer(model=model, args=train_args, processing_class=tokenizer, train_dataset=train_ds)
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_training_meta(
        output_dir,
        org_id=args.org_id,
        base_model=args.model_id,
        dataset_path=dataset_path,
        dataset_hash=args.dataset_hash,
        init_adapter=args.init_adapter,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

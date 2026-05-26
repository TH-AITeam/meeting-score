"""学習済み LoRA の検証（ホールドアウト val.jsonl での予測精度）。

ベース Qwen3.5-9B + 学習済み LoRA アダプタを Unsloth で読み込み、
val.jsonl の user プロンプトを推論させ、出力 JSON を gold ラベルと比較する。

測る指標:
  - JSON パース成功率 / スキーマ妥当率
  - speech_type 正答率
  - スコア7軸・ペナルティ4軸の MAE と完全一致率
  - 合計スコア（scores 合計 + penalties 合計）の Pearson 相関 / MAE

使い方:
  source scripts/lora/.venv-lora/bin/activate
  python scripts/lora/eval_lora.py --limit 60        # まず一部で確認
  python scripts/lora/eval_lora.py                   # val 全件
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_lora")

REPO_ROOT = Path(__file__).resolve().parents[2]

SCORE_AXES = [
    "issue_clarification", "decision_progress", "risk_detection",
    "actionability", "groundedness", "novelty", "summarization",
]
PENALTY_AXES = ["duplication", "verbosity", "off_topic", "unsupported_assertion"]
SPEECH_TYPES = {
    "論点整理", "提案", "質問", "情報共有", "要約",
    "懸念提示", "根拠提示", "意思決定促進", "雑談/脱線",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="学習済み LoRA をホールドアウトで検証")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    p.add_argument("--adapter", default=str(REPO_ROOT / "outputs" / "qwen35-9b-lora"))
    p.add_argument("--val", default=str(REPO_ROOT / "data" / "annotations" / "kokkai" / "distill" / "val.jsonl"))
    p.add_argument("--max-seq-len", type=int, default=6144)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ評価（0=全件）")
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "qwen35-9b-lora" / "eval_predictions.jsonl"))
    p.add_argument("--no-adapter", action="store_true",
                   help="LoRA を載せず素のベースモデル（--model-id）を評価する")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="大きいモデルを 4bit でロードして VRAM を節約する")
    p.add_argument("--backend", choices=["unsloth", "transformers"], default="unsloth",
                   help="モデルのロード経路。NVFP4/MoE 等 unsloth 非対応なら transformers")
    p.add_argument("--gpu-mem-gib", type=int, default=24,
                   help="transformers backend で GPU に載せる重みの上限(GiB)。残りはCPUへオフロード")
    p.add_argument("--device-map", choices=["auto", "cuda"], default="auto",
                   help="transformers backend の配置。cuda=全GPU(高速・要VRAM)、auto=溢れをCPUオフロード(低速)")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.set_defaults(enable_thinking=False)
    return p.parse_args()


def extract_json(text: str) -> dict | None:
    """生成テキストから最初の JSON オブジェクトを取り出す。"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def schema_ok(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("speech_type") not in SPEECH_TYPES:
        return False
    sc, pen = obj.get("scores"), obj.get("penalties")
    if not isinstance(sc, dict) or not isinstance(pen, dict):
        return False
    for a in SCORE_AXES:
        if clamp(sc.get(a), 0, 3) is None or not (0 <= float(sc.get(a, -1)) <= 3):
            return False
    for a in PENALTY_AXES:
        if clamp(pen.get(a), -3, 0) is None or not (-3 <= float(pen.get(a, 1)) <= 0):
            return False
    return True


def total_of(obj: dict) -> float:
    s = sum(clamp(obj.get("scores", {}).get(a), 0, 3) or 0 for a in SCORE_AXES)
    p = sum(clamp(obj.get("penalties", {}).get(a), -3, 0) or 0 for a in PENALTY_AXES)
    return float(s + p)


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main() -> None:
    args = parse_args()

    rows = [json.loads(l) for l in Path(args.val).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    logger.info("検証件数: %d", len(rows))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"検証結果  (n=0, adapter={args.adapter})")
        print("評価対象が 0 件のため、指標計算をスキップしました。")
        print("=" * 60)
        print(f"予測の詳細を書き出し: {args.out}")
        return

    import torch

    # --no-adapter なら素のベース（--model-id）、それ以外は学習済みアダプタを読む。
    model_name = args.model_id if args.no_adapter else args.adapter
    logger.info("ロード対象: %s (backend=%s, 4bit=%s)", model_name, args.backend, args.load_in_4bit)
    if args.backend == "transformers":
        # NVFP4/ハイブリッド MoE 等、unsloth のコンパイル済みモジュールで
        # 動かないモデル向け。量子化設定はモデルの config から自動適用される
        # （compressed-tensors / AWQ 等）。--no-adapter 前提（素ベース評価）。
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import AutoProcessor
            tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # NVFP4 35B はGPU(32GB)に重み＋KVキャッシュが収まらないため、
        # GPU上限を絞って溢れを accelerate が CPU へオフロードする
        # （活性化/KV 用に GPU を ~7GB 空ける）。オフロード分は遅いが動く。
        load_kwargs = dict(torch_dtype="auto", device_map=args.device_map, trust_remote_code=True)
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if args.device_map == "auto":
            load_kwargs["max_memory"] = {0: f"{args.gpu_mem_gib}GiB", "cpu": "120GiB"}
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # adapter_config が base を指す（アダプタ時）
            max_seq_length=args.max_seq_len,
            dtype=torch.bfloat16,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=False,
        )
        FastLanguageModel.for_inference(model)
    # Qwen3.5-9B は VL（マルチモーダル）モデルで from_pretrained は Processor を返す。
    # Processor を直接呼ぶと第1引数が images 扱いになり、テキストを画像として
    # 解釈してしまうため、内部の純テキスト tokenizer を取り出して使う。
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    # device_map でオフロードされたモデルでも入力は埋め込み層のあるデバイスへ。
    input_device = getattr(model, "device", None) or next(model.parameters()).device

    n_json = n_schema = n_speech = 0
    abs_err = {a: [] for a in SCORE_AXES + PENALTY_AXES}
    exact = {a: 0 for a in SCORE_AXES + PENALTY_AXES}
    pred_totals, gold_totals, total_abs = [], [], []
    out_f = out_path.open("w", encoding="utf-8")

    for i, row in enumerate(rows):
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        gold = json.loads(next(m for m in row["messages"] if m["role"] == "assistant")["content"])

        try:
            text = tokenizer.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True,
            )
        inputs = text_tok(text, return_tensors="pt").to(input_device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=text_tok.eos_token_id,
            )
        gen = text_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_json(gen)

        rec = {"idx": i, "gold": gold, "raw": gen}
        if pred is not None:
            n_json += 1
            rec["pred"] = pred
            if schema_ok(pred):
                n_schema += 1
            if pred.get("speech_type") == gold.get("speech_type"):
                n_speech += 1
            for a in SCORE_AXES:
                pv, gv = clamp(pred.get("scores", {}).get(a), 0, 3), gold["scores"][a]
                if pv is not None:
                    abs_err[a].append(abs(pv - gv))
                    exact[a] += int(pv == gv)
            for a in PENALTY_AXES:
                pv, gv = clamp(pred.get("penalties", {}).get(a), -3, 0), gold["penalties"][a]
                if pv is not None:
                    abs_err[a].append(abs(pv - gv))
                    exact[a] += int(pv == gv)
            pt = total_of(pred)
            gt = total_of(gold)
            pred_totals.append(pt); gold_totals.append(gt); total_abs.append(abs(pt - gt))
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d 完了 (JSON成功 %d)", i + 1, len(rows), n_json)

    out_f.close()
    n = len(rows)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 60)
    print(f"検証結果  (n={n}, adapter={args.adapter})")
    print("=" * 60)
    print(f"JSON パース成功率 : {n_json}/{n} = {n_json/n:.1%}")
    print(f"スキーマ妥当率    : {n_schema}/{n} = {n_schema/n:.1%}")
    print(f"speech_type 正答率: {n_speech}/{n_json} = {n_speech/max(n_json,1):.1%} (JSON成功分)")
    print("-" * 60)
    print(f"{'軸':<22}{'MAE':>8}{'完全一致':>10}")
    for a in SCORE_AXES:
        print(f"{a:<22}{mean(abs_err[a]):>8.3f}{exact[a]/max(n_json,1):>9.1%}")
    print("-- penalties --")
    for a in PENALTY_AXES:
        print(f"{a:<22}{mean(abs_err[a]):>8.3f}{exact[a]/max(n_json,1):>9.1%}")
    print("-" * 60)
    all_score_mae = mean([e for a in SCORE_AXES for e in abs_err[a]])
    all_pen_mae = mean([e for a in PENALTY_AXES for e in abs_err[a]])
    print(f"スコア7軸 平均MAE : {all_score_mae:.3f}")
    print(f"ペナルティ平均MAE : {all_pen_mae:.3f}")
    print(f"合計スコア MAE    : {mean(total_abs):.3f}  (合計スコア理論域 -12〜21)")
    print(f"合計スコア相関 r  : {pearson(pred_totals, gold_totals):.3f}")
    print("=" * 60)
    print(f"予測の詳細を書き出し: {args.out}")


if __name__ == "__main__":
    main()

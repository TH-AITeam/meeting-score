from datasets import load_dataset
import soundfile as sf
from pathlib import Path

out_dir = Path.home() / "mtg-score/meeting-score/data/eval_audio/meeting_01"
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "reazon-research/reazonspeech",
    "tiny",
    split="train",
    trust_remote_code=True
)

sample = ds[0]
print("sample keys:", sample.keys())

audio = sample["audio"]
sf.write(out_dir / "audio.wav", audio["array"], audio["sampling_rate"])

text = (
    sample.get("transcription")
    or sample.get("text")
    or sample.get("sentence")
)

if text is None:
    raise ValueError(f"文字起こしカラムが見つかりません: {sample.keys()}")

(out_dir / "reference.txt").write_text(text.strip() + "\n", encoding="utf-8")

print("created:")
print(out_dir / "audio.wav")
print(out_dir / "reference.txt")
print("reference:", text)

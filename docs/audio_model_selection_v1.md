# 音声処理モデル選定 v1 (Issue #19)

候補比較の総合スコアシート。実測値は SSH 先 (RTX 5090 32GB) で
`scripts/run_audio_benchmark.sh` を回した結果を転記する。

| 環境項目 | 値 |
|---|---|
| 実行ホスト | SSH 先 RTX 5090 (32GB VRAM, sm_120 / Blackwell) |
| ASR 実装 | WhisperX (https://github.com/m-bain/whisperX) |
| Diarization 実装 | pyannote.audio 3.x |
| 補助 | librosa による音量分析 |
| 評価データ | `data/eval_audio/` 最低 3 本 (自作録音 or 合成音声、本物の会議は不使用) |
| 評価日 | TBD（実測完了後に更新） |

## ASR スコアシート

| モデル (backbone) | 量子化 | VRAM (GB) | CER (%) | 固有名詞認識率 (%) | RTF (1h 音声 / 処理時間) | word-ts 精度 (人手評価) | ライセンス |
|---|---|---|---|---|---|---|---|
| WhisperX (whisper-large-v3) | fp16 | TBD | TBD | TBD | TBD | TBD | BSD-2 + MIT |
| WhisperX (kotoba-whisper-v2.0) | fp16 | TBD | TBD | TBD | TBD | TBD | BSD-2 + Apache 2.0 |
| faster-whisper (large-v3) | int8_float16 | TBD | TBD | TBD | TBD | TBD | MIT |
| Reazon Speech v2 (NeMo) | fp16 | TBD | TBD | TBD | TBD | TBD | Apache 2.0 |

数値の出典: `reports/audio_benchmarks/{model_id}/{timestamp}_asr.json`

## Diarization スコアシート

| モデル | DER (%) | オーバーラップ accuracy (%) | 話者数推定精度 (人手評価) | ライセンス |
|---|---|---|---|---|
| pyannote/speaker-diarization-3.1 | TBD | TBD | TBD | MIT (lib) + 利用規約 |
| NVIDIA NeMo Diarization | TBD | TBD | TBD | Apache 2.0 |

数値の出典: `reports/audio_benchmarks/{model_id}/{timestamp}_diar.json`

## 統合パイプライン スコアシート

ASR × Diarization の組み合わせで、最終的に「正しい発言単位に区切れた率」を人手評価する。

| ASR | Diarization | 統合発言分割精度 (%) | 平均発言粒度 (秒) | コメント |
|---|---|---|---|---|
| WhisperX (large-v3) | pyannote-3.1 | TBD | TBD | 第一採用想定の組合せ |
| WhisperX (kotoba) | pyannote-3.1 | TBD | TBD | 日本語 finetune backbone への差替検証 |
| WhisperX (large-v3) | NeMo | TBD | TBD | Diar を NeMo に振った場合 |

数値の出典: `reports/audio_benchmarks/{asr_id}_x_{diar_id}/{timestamp}_pipeline.json`

## 軸重み（採用判定用）

| 軸 | 重み |
|---|---|
| 統合発言分割精度（最終アウトカム） | **0.35** |
| ASR CER | 0.20 |
| Diar DER | 0.20 |
| 固有名詞認識率 | 0.10 |
| RTF (小さいほど良) | 0.05 |
| オーバーラップ accuracy | 0.05 |
| ライセンス自由度 | 0.05 |

実測が揃った段階で min-max 正規化 → 重み付き合計でランキングを出す。

## 暫定採用（実測前の予想）

- **ASR 第一採用**: **WhisperX (whisper-large-v3 backbone)**
  - VAD 内蔵 / word-level timestamp の精度実績 / 日本語 OK
  - backbone は実測後に Kotoba Whisper への差替え検討
- **Diarization 第一採用**: **pyannote/speaker-diarization-3.1**
  - overlap segmentation が標準で使える、HF token 確認
- **音量分析**: librosa の RMS / short-time energy / ZCR を `app/asr/volume_analyzer.py` に実装し、`Utterance.volume_level` を補強情報として付与

**最終採用は実測後に確定し、本ドキュメントの数値列を埋めて ADR Status を Accepted に切替える。**

## 実測手順

SSH 先 (5090) で:

```bash
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score
cd backend && uv venv --python 3.13 --clear && cd ..
source backend/.venv/bin/activate

# audio extras を入れる (WhisperX / pyannote / librosa)
uv sync --extra audio

# HF token を環境変数に
export HUGGINGFACE_HUB_TOKEN=hf_xxxxx   # pyannote の gated repo アクセス用

# data/eval_audio/ にローカルで音声 + reference を用意（data/eval_audio/README.md 参照）

bash scripts/run_audio_benchmark.sh --all 2>&1 | tee /tmp/issue19_audio_bench.log
```

集計は本ドキュメントの各表に手動転記する（PR にコミットしてエビデンスとして残す）。

## 採用後にやること（受け入れ条件のチェックリスト）

- [ ] ASR / Diarization の各表が TBD 無しで埋まる
- [ ] 統合パイプラインの想定品質 (CER, DER) が `Readme.md` に書かれる
- [ ] 採用モデルを `backend/config.yaml.example` の `audio:` セクションに固定
- [ ] `docs/adr/0002-audio-model.md` の Status を `Proposed` → `Accepted`
- [ ] `docs/audio_model_history.md` の v1 採用日を確定
- [ ] Issue #11 (音声入力パイプライン本実装) のコメントで採用モデルを共有

## 関連

- Issue #19 (本 Issue)
- Issue #11: 音声入力パイプライン本実装（採用モデルを `app/asr/` 経由で参照）
- `docs/audio_model_candidates.md`: 候補絞り込み詳細
- `docs/adr/0002-audio-model.md`: 意思決定の根拠記録

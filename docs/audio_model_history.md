# 音声処理モデル採用履歴

ASR + 話者分離 + 音量分析 の世代管理。判断 LLM (`docs/model_history.md`) と同じ運用で、半年〜1年で見直す前提で履歴を残す。

## 採用履歴

| 世代 | 採用日 | ASR | Diarization | 補助 | 判断軸の重点 | ADR | 廃止日 |
|---|---|---|---|---|---|---|---|
| v1 | 2026-05-12 (Proposed) | WhisperX (`openai/whisper-large-v3`) | `pyannote/speaker-diarization-3.1` | librosa (RMS / short-time energy / ZCR) | word-level timestamp + 日本語実用品質 + overlap 対応 | [ADR 0002](adr/0002-audio-model.md) | – |

> 採用日は ADR が Proposed のときの日付。**Accepted に切り替わったらここを上書きする**。

## 見直しトリガ

- **時期**: 半年〜1年ごとに棚卸し
- **イベント駆動**:
  - Whisper の新世代 (v4 など) が出て CER が有意に下がった
  - Kotoba Whisper など日本語 finetune backbone が WhisperX に差し替え可能で large-v3 を上回った
  - pyannote 4.x など Diarization の世代が変わった
  - 採用モデルの推論速度 (RTF) が会議処理要件を満たさなくなった
  - ライセンス変更で再配布／蒸留が制約された
  - 多言語 (日英混在) 会議を対象にする時 (別 ADR)

## 更新手順

1. 新候補を `docs/audio_model_candidates.md` に追加（必要なら全面差し替え）
2. `data/eval_audio/` のセットを v1 から流用、または更新
3. `scripts/run_audio_benchmark.sh --all` で実測
4. `docs/audio_model_selection_v1.md` を `audio_model_selection_v{N}.md` にコピー → 更新
5. ADR を追加 (`docs/adr/000N-audio-model.md`)
6. 本ファイルの表に新世代を追加し、旧世代の「廃止日」を記入
7. `backend/config.yaml.example` の `audio:` セクションを更新

## 関連

- `docs/adr/`: 個別の意思決定記録
- `docs/audio_model_candidates.md`: 現世代の候補絞り込み詳細
- `docs/audio_model_selection_v{N}.md`: 世代ごとの総合スコアシート
- Issue #19: 初回世代 v1 の選定
- 判断 LLM 側の世代管理: `docs/model_history.md`

# ADR 0002: 音声処理モデル (ASR + 話者分離) の採用

- **Status**: Proposed（実測後に Accepted に更新予定）
- **Date**: 2026-05-12
- **Deciders**: TomokiAkiyama06
- **Related**: Issue #19, Issue #11, Issue #5, ADR 0001 (#18 判断 LLM)

## Context

会議貢献度スコアリングの入力として「発言単位 + 話者ラベル + テキスト + word-level timestamp」の JSON を作る必要がある。これは Issue #11 (`app/asr/` 実装) の前提となるモデル選定。実行環境は **RTX 5090 32GB (Blackwell)**、開発は Mac で、実機評価は SSH 先で行う。

主要な要件:
- 日本語実用品質の ASR
- word-level timestamp が取れる (発言粒度の境界判定に必要)
- 2〜6 人の会議で話者を分離できる (オーバーラップ発話に対応)
- 商用利用可・OSS

## Decision

第一採用 (Proposed) は以下の組合せ:

| 役割 | モデル / ライブラリ |
|---|---|
| **ASR** | **WhisperX** (`openai/whisper-large-v3` を faster-whisper backend で利用) |
| **Diarization** | **pyannote/speaker-diarization-3.1** (pyannote.audio 3.x) |
| **補助 (音量分析)** | **librosa** + numpy (RMS / short-time energy / ZCR) |

`docs/audio_model_selection_v1.md` のスコアシート参照。**Status は Proposed**。CER / RTF / DER / 固有名詞認識率の実測が完了し、第一採用組合せが確定した段階で Accepted に切り替える。本 PR は実測 TBD のためマージしても Issue #19 を close しない（`Refs #19`）。

## Options Considered

`docs/audio_model_candidates.md` で詳述した候補:

ASR:
1. **WhisperX** ← 第一候補 (word-level timestamp + 日本語 + VAD 内蔵)
2. **faster-whisper** 単体 — word-level の精度が WhisperX 経由より劣る
3. **Kotoba Whisper v2.0** — backbone 差替候補。WhisperX の model_dir を差し替えて検証可
4. **Reazon Speech v2 (NeMo)** — NeMo 依存で deploy が重い

Diarization:
1. **pyannote/speaker-diarization-3.1** ← 第一候補 (overlap segmentation 対応)
2. **NVIDIA NeMo Diarization** — pyannote と精度差が決定的でない限り deploy 負担を負う必要なし
3. **WeSpeaker / 3D-Speaker** — clustering/VAD を別途組む必要があり成熟度に劣る

## Rationale

### WhisperX を第一候補にする理由

1. **word-level timestamp が必須要件**: 仕様 #19 に「ASR の word-ts と突き合わせられる出力形式」と明記。WhisperX は wav2vec2 アライメントを内部で使い word 単位の正確な timestamp を出す
2. **日本語実用品質**: Whisper large-v3 は JSUT / CSJ などの日本語ベンチで CER 10〜15% 水準。会議で議論を成立させるのに十分
3. **faster-whisper backend で 5090 上で高速**: RTF 0.05〜0.15 を見込め、1 時間音声を 3〜10 分で処理
4. **VAD 内蔵 + pyannote と統合実装が整備**: WhisperX 本体が pyannote の VAD と diarization を統合実行する例を提供
5. **backbone 差替えが容易**: Kotoba Whisper など日本語 finetune が伸びてきたら model_dir を差し替えるだけで適用可能（実測で large-v3 を上回ったら採用）

### pyannote-3.1 を第一候補にする理由

1. **デファクト**: 多数の研究 / 商用採用実績。検証データの再現性が高い
2. **オーバーラップ発話対応**: 3.x で `PowersetMultilabel` による同時発話のラベル付けが可能。会議では「被せて喋る」が起きるので必須
3. **WhisperX との統合**: word-level timestamp を持つ ASR と speaker label を結合する処理が標準パターンとして確立
4. **HF 配布**: token 取得さえできれば pip + HUGGINGFACE_HUB_TOKEN で展開可

### 音量分析を入れる理由

- ASR + Diar だけでは「強い発言／ぼそっと言った発言」の区別がつかない。会議の温度感をスコアリングに反映できる余地を残しておきたい
- pyannote の overlap_detection だけでは「2 人とも小さい声で重なった」のような重なりを取り損ねるケースがあり、RMS で補える
- 実装は librosa + numpy で 30 行程度。コストが低い

## Consequences

採用 (Accepted) 確定後:

- `backend/config.yaml.example` の `audio:` セクションに WhisperX のモデル名 / pyannote のモデル名 / HF token 取得方法を固定
- Issue #11 (`app/asr/`) は本 PR の `app/asr/base.py` Protocol を実装する形で着手可能
- `docs/audio_model_history.md` に v1 採用エントリを追加
- 音声依存は `pyproject.toml` の `[project.optional-dependencies] audio = [...]` で隔離（CI を重くしないため、`uv sync --extra audio` で明示インストール）

将来の見直しトリガ:

- 半年〜1年に 1 回の棚卸し
- Kotoba Whisper v3 / Whisper v4 など backbone の新世代が出た時
- pyannote 4.x がリリースされた時
- 多言語 (日英混在) 会議を本格対象にする時 → 別 ADR で再評価

## Open Questions

- [ ] **CER / RTF / DER の実測**: SSH 先 (5090) で `scripts/run_audio_benchmark.sh --all` を実行し、`docs/audio_model_selection_v1.md` の TBD 列を埋める
- [ ] **HF token 配布の運用**: pyannote/speaker-diarization-3.1 は gated repo。チーム共有時の運用 (個人 token / 組織 token) を `data/eval_audio/README.md` に記載
- [ ] **オーバーラップ評価の方法**: 公開コーパスに overlap ラベル付きデータが少ないため、自前録音で人手アノテする工数を見積もる
- [ ] **音量分析の閾値**: RMS の「low/mid/high」の境界を固定値で持つか、録音ごとに正規化するかを実測後に決定

## References

- `docs/audio_model_candidates.md`: 候補絞り込みの詳細
- `docs/audio_model_selection_v1.md`: 総合スコアシート (実測転記先)
- `data/eval_audio/README.md`: 評価音声の用意手順とプライバシー指針
- Issue #19: 本 Issue
- Issue #11: 採用モデルを参照する本実装

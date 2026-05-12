# 音声処理モデル候補リスト (Issue #19)

会議の音声を「発言単位 + 話者ラベル + テキスト + word-level timestamp」へ変換するための ASR / 話者分離モデルの候補を整理する。採用方針は **WhisperX (ASR) + pyannote-audio (Diarization) + librosa による音量分析 (補助)** の組み合わせ。

## 共通制約

| 項目 | 値 |
|---|---|
| 実行 GPU | NVIDIA RTX 5090 (32GB VRAM, sm_120 / Blackwell) |
| ライセンス | 商用利用可・OSS |
| 出力形式 | 各発言に `start_sec` / `end_sec` / `speaker` / `text` / `words[]` を持つ JSON |
| 言語 | 日本語が主 (#19 注意の「多言語(日英混在)」は v1 では非ゴール) |
| 想定入力 | 2〜6 人の会議、15 分〜90 分、wav/mp3/m4a |

## A) ASR 候補

### 1. WhisperX (採用第一候補)

| 項目 | 値 |
|---|---|
| 実装 | https://github.com/m-bain/whisperX |
| バックボーン | `openai/whisper-large-v3` (faster-whisper backend で高速化) |
| ライセンス | BSD-2 (whisperX) + MIT (Whisper 本体) |
| word-level timestamp | ◎ (内部に wav2vec2.0 アライメント) |
| 日本語精度 | 良 (Whisper large-v3 が JSUT / CSJ で CER ~10〜15% 水準、ドメイン依存) |
| VAD | 内蔵 (pyannote VAD を任意で使える) |
| RTF 目安 (5090 / large-v3) | 0.05〜0.15 (1 時間音声を 3〜10 分で処理) |
| 採用根拠 | (a) word-level timestamp が必須要件、(b) Diarization と timestamp で合わせるための anchor、(c) VAD と silence の同時取得、(d) 派生 finetune (Kotoba Whisper 等) を backbone 差し替えで採用可能 |

### 2. faster-whisper (落選候補・参考)

| 項目 | 値 |
|---|---|
| HF / 実装 | https://github.com/SYSTRAN/faster-whisper |
| ライセンス | MIT |
| word-level timestamp | △ (Whisper 自身の token timestamp、揺れあり) |
| 採用しない理由 | word-level の精度が WhisperX 経由より劣る。WhisperX の内部で結局使うので、独立採用しない |

### 3. Kotoba Whisper v2.0 (将来差替候補)

| 項目 | 値 |
|---|---|
| HF | `kotoba-tech/kotoba-whisper-v2.0` |
| ライセンス | Apache 2.0 |
| 日本語精度 | ◎ (日本語 finetune、CER は Whisper large-v3 を上回るレポートあり) |
| word-level timestamp | WhisperX backbone に差し替えれば取得可能 |
| 採用しない理由 (v1) | WhisperX 本体は Whisper large-v3 で十分検証されている。Kotoba Whisper は実測で large-v3 を有意に上回った場合に backbone を差し替える形で採用 |

### 4. Reazon Speech v2 (落選)

| 項目 | 値 |
|---|---|
| HF | `reazon-research/reazonspeech-nemo-v2` |
| ライセンス | Apache 2.0 |
| 日本語精度 | ◎ (日本語特化、CSJ で CER 良好) |
| word-level timestamp | △ (NeMo の token timestamp は粒度粗め) |
| 採用しない理由 | NeMo 依存で deploy が重い。WhisperX のような wav2vec2 アライメントとの組み合わせ実装が未整備 |

## B) 話者分離 (Diarization) 候補

### 1. pyannote/speaker-diarization-3.1 (採用第一候補)

| 項目 | 値 |
|---|---|
| HF | `pyannote/speaker-diarization-3.1` (gated repo、HF token 必須) |
| 実装ライブラリ | https://github.com/pyannote/pyannote-audio |
| ライセンス | MIT (lib) / model はユーザ規約あり (商用可) |
| 話者数自動検出 | ✓ / `num_speakers` で固定も可 |
| オーバーラップ発話 | ✓ (`PowersetMultilabel` で同時発話を扱える) |
| 採用根拠 | (a) 事実上のデファクトで検証実績が最大、(b) WhisperX との統合例が公開されている、(c) overlap segmentation で「被せ発話」を別話者として分離 |

### 2. NVIDIA NeMo Speaker Diarization (落選)

| 項目 | 値 |
|---|---|
| HF / 実装 | `nvidia/diar_sortformer_4spk-v1` 等 |
| ライセンス | Apache 2.0 |
| 採用しない理由 | NeMo / Triton 依存で deploy が重い。pyannote と精度差が決定的でない限り採用しない |

### 3. WeSpeaker / 3D-Speaker (落選)

| 項目 | 値 |
|---|---|
| 採用しない理由 | speaker embedding は強いが clustering / VAD 部分を別途組む必要があり、`pyannote` のパイプライン完成度に劣る |

## C) 音量分析 (補助)

「モデル」ではなく numpy / librosa の信号処理で実装する補強ロジック。pyannote の overlap_detection と組み合わせる。

| 役割 | 計算 | 用途 |
|---|---|---|
| RMS energy | `librosa.feature.rms` | 発話セグメントごとの平均音量レベル → `volume_level: low/mid/high` |
| Short-time energy | numpy で 25ms 窓 hop 10ms | 無音区間の検出（pyannote VAD の二重確認） |
| Zero-crossing rate | `librosa.feature.zero_crossing_rate` | 摩擦音・笑い声などの非音声判定 |

採用根拠:
- 同じ発言でも「強く言った／ぼそっと言った」で会議の温度を出すための情報を残す
- pyannote の overlap だけでは「2 人とも小さい声」のような重なりを取り損ねるケースを RMS で補える
- 既存実装が librosa で 30 行程度で書けて軽量

## 評価軸（Issue #19 仕様より再掲）

| カテゴリ | 軸 | 計測方法 |
|---|---|---|
| ASR | CER (文字誤り率) | 自前録音3本 + 公開コーパスで測定 |
| ASR | RTF (real-time factor) | 5090 で wall-clock |
| ASR | 固有名詞耐性 | 会議特有用語 (プロダクト名/人名) の認識率 |
| Diar | DER (Diarization Error Rate) | 同上 |
| Diar | オーバーラップ発話処理 | 被せ発話を分割できるか |
| 統合 | 結合後の発言分割精度 | 「正しい発言単位に区切れた率」を人手評価 |

## 評価データの扱い

- 評価音声は `data/eval_audio/` に最低 3 本（会議形式、2〜4 人、各 15 分以上）置く
- **本物の会議は使わない** (プライバシー)。自作録音または合成音声 (TTS 多話者) を使う
- 詳細は `data/eval_audio/README.md` 参照

## 関連

- Issue #19 (本 Issue)
- Issue #11: 音声入力パイプライン本実装（採用モデルを `app/asr/` 経由で参照する）
- `docs/audio_model_selection_v1.md`: 総合スコアシート（実測転記先）
- `docs/adr/0002-audio-model.md`: 意思決定の根拠記録

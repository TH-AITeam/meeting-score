# #10 [feature] 音声入力パイプライン(ASR + 話者分離)

**Labels**: `feature`, `breaking-change`, `P1`
**Milestone**: v0.4

## 前提

**先に #18 でモデル選定を済ませること**。 ここでは「採用 ASR」「採用 Diar」と抽象的に書く。

## 概要

音声ファイル(wav / mp3 / m4a) → 既存の `MeetingInput` JSON への変換パイプラインを追加。AGENT.md は当初これを範囲外にしていたが、現実の入口を作る。

## パイプライン構造

```
[audio] → [ASR(採用モデル)] → [word-level timestamps]
                                  ↓
                              [Diar(採用モデル)] ←──────┐
                                  ↓                    │
                              [話者付き発言列]         │
                                  ↓                    │
                              [メタ抽出(LLM, #17)] ─→ MeetingInput JSON
```

## モジュール構成

```
app/asr/
  __init__.py
  transcribe.py        # ASR ラッパ(採用モデルを差し替え可能に)
  diarize.py           # Diar ラッパ(採用モデルを差し替え可能に)
  segmenter.py         # word-ts + diar 結果を発言単位にマージ
  meeting_builder.py   # → MeetingInput JSON
prompts/
  meta_extraction.txt  # title / goal / agenda をテキスト全文から LLM で抽出
```

## やること

- [ ] `app/asr/transcribe.py`: 採用 ASR のラッパ。word-level timestamp 取得まで
- [ ] `app/asr/diarize.py`: 採用 Diar のラッパ
- [ ] `app/asr/segmenter.py`: word-ts と話者を突き合わせ、発言単位に正規化
  - 同一話者連続の発話は3秒未満の無音まで1発言として結合
- [ ] `app/asr/meeting_builder.py`:
  - title / goal / agenda / decision_points を LLM 1コール(#17 の採用モデル)で抽出
  - utterance_id を時系列で `u001`〜 採番
- [ ] CLI: `python -m app.asr.cli --input audio.wav --output meeting.json`
- [ ] FastAPI に `/upload_audio` エンドポイント追加(任意)
- [ ] サンプル音声を `data/sample_audio/` に追加(プライバシー注意、自分で録音 or 合成音声)

## 完了条件

- 30分の会議音声で `MeetingInput` JSON が自動生成される
- 生成された JSON で既存パイプラインがそのまま動く
- 採用モデルの精度指標(CER / DER) を README に記載

## リスク

- 話者分離は人数が多いほど劣化する。本番運用では手動補正手段(GUI / JSON 編集)も用意
- ASR の認識誤りはそのまま評価に伝播するので、誤字耐性のあるプロンプト設計も忘れずに
- 採用モデルのライセンス・トークン要件は #18 で必ず確認

## 関連

- 採用モデルは #18 で決定
- メタ抽出 LLM は #17 で決定したローカルモデル(#11 が立った後)

# 会議貢献度スコアリング

会議の文字起こし JSON を読み込み、各発言が会議をどれだけ前に進めたかを評価・可視化する MVP です。
人の優劣ではなく、会議目的への実質的な貢献を振り返るためのツールとして扱います。

## 機能

- 会議ログ JSON の読み込み
- 各発言の発言タイプ、軸別スコア、減点、総合スコア、理由の算出
- 会議全体のサマリー
- 会議を前に進めた発言の抽出
- 話者別の発言傾向サマリー
- JSONアップロードとサンプルデータ分析に対応したWeb UI

## 入力形式

```json
{
  "meeting_id": "m001",
  "title": "新機能企画会議",
  "goal": "初回リリース範囲を決める",
  "agenda": ["対象ユーザー確認", "リリース範囲整理", "スケジュール確認"],
  "decision_points": ["初回に含める機能", "見積もり前提条件"],
  "utterances": [
    {
      "utterance_id": "u001",
      "speaker": "A",
      "timestamp": "00:12:31",
      "text": "今日決めるべき範囲を先に確認しましょう。"
    }
  ]
}
```

`utterance_id`, `speaker`, `timestamp`, `text` が欠けている発言は、読み込み時に安全な既定値で補完されます。

## 評価軸

加点軸は 0 から 3、減点軸は -3 から 0 で評価します。

- 論点整理
- 意思決定促進
- リスク検知
- アクション化
- 根拠性
- 新規性
- 要約/交通整理
- 重複
- 冗長さ
- 脱線
- 根拠の薄い断言

## セットアップ

```bash
uv sync
```

`.env` に OpenAI API キーを設定します。

```env
OPENAI_API_KEY=your_api_key_here
```

モデル名は `config.yaml` の `llm.model` で変更できます。既定値は `gpt-5.4-mini` です。

## 実行

```bash
uv run python run.py
```

起動後、ブラウザで `http://localhost:8000` を開きます。

## テスト

```bash
uv run pytest -q
```

## 構成

- `app/api`: FastAPI ルートとアプリケーション設定
- `app/services`: 分析パイプライン
- `app/ingest`: 入力データの読み込みと正規化
- `app/context_builder`: 発言評価に使う前後文脈の生成
- `app/evaluators`: LLM 評価
- `app/scoring`: スコア計算とルール補正
- `app/aggregation`: 会議・話者単位の集計
- `app/reporting`: UI向けレスポンス整形
- `ui`: 静的Web UI

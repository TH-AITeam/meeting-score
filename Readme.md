# README

[![CI](https://github.com/TH-AITeam/meeting-score/actions/workflows/ci.yml/badge.svg)](https://github.com/TH-AITeam/meeting-score/actions/workflows/ci.yml)

## 概要

このプロジェクトは、会議中の各発言について「会議をどれだけ前進させたか」を複数軸で評価し、会議改善・振り返りに使える形で可視化する MVP を実装するためのものです。

評価対象は、**人の優秀さそのものではなく、会議への実質的な貢献**です。
そのため、本プロジェクトは人事査定や社員ランキングを目的としません。

---

# 何をするツールか

会議の文字起こしデータを入力すると、各発言に対して以下を出力します。

* 発言タイプ
* 軸別スコア
* 減点判定
* 総合貢献スコア
* 理由

さらに、会議全体および話者単位で集計し、以下を可視化します。

* 会議を前進させた発言 Top 5
* 論点整理に貢献した発言
* 意思決定を進めた発言
* リスクを可視化した発言
* アクション化した発言
* 話者別サマリー
* 会議改善コメント

---

# 目的

この MVP の主目的は次の通りです。

* 明らかに良い発言を拾えること
* 単に長く話した人が勝たないこと
* 論点整理、要約、リスク指摘のような見えにくい貢献を可視化できること
* 会議終了後の振り返りに使えること

---

# やらないこと

MVP では以下を対象外にします。

* 人事評価
* 昇進査定連携
* リアルタイム会議アシスト
* 高度な教師あり学習
* すべての会議タイプへの対応

---

# 対象会議

最初は以下のいずれか1種類に絞る前提です。

* 企画会議
* 開発定例
* 要件定義会議

推奨初期対象は **企画会議** または **開発定例** です。

ブレインストーミングや 1on1 は、評価基準がぶれやすいため後回しにします。

---

# 設計思想

このプロジェクトでは、以下を重視します。

1. 発言量は加点しない
2. 長く話しただけでは高評価にしない
3. 短くても会議を動かした発言を高く評価する
4. 肩書きや立場で補正しない
5. 既出内容の繰り返しは減点する
6. 会議目的への寄与で評価する
7. UI を人物ランキングっぽくしない

---

# 入力

## 必須入力

* 会議タイトル
* 会議目的
* 発言列（発言単位に分割済み）
* 話者
* タイムスタンプ

## 推奨入力

* アジェンダ
* 決めるべき事項
* 現在の議題
* 会議終了時の決定事項

## 入力例

```json
{
  "meeting_id": "m001",
  "title": "新機能企画会議",
  "goal": "初回リリース範囲を決める",
  "agenda": ["対象ユーザー確認", "初回リリース範囲整理", "スケジュール確認"],
  "decision_points": ["初回に含める機能", "見積もり前提条件"],
  "utterances": [
    {
      "utterance_id": "u001",
      "speaker": "A",
      "timestamp": "00:12:31",
      "text": "今決めるべきはUI仕様ではなく、今月中に出す範囲ですよね。"
    }
  ]
}
```

---

# 出力

各発言について、最低限以下を返します。

* `speech_type`
* `scores`
* `penalties`
* `total_score`
* `reason`

## 出力例

```json
{
  "utterance_id": "u001",
  "speaker": "A",
  "timestamp": "00:12:31",
  "text": "今決めるべきはUI仕様ではなく、今月中に出す範囲ですよね。",
  "speech_type": "論点整理",
  "scores": {
    "issue_clarification": 3,
    "decision_progress": 2,
    "risk_detection": 0,
    "actionability": 0,
    "groundedness": 1,
    "novelty": 2,
    "summarization": 1
  },
  "penalties": {
    "duplication": 0,
    "verbosity": 0,
    "off_topic": 0,
    "unsupported_assertion": 0
  },
  "total_score": 6.6,
  "reason": "議論の焦点をUI詳細からリリース範囲に戻し、意思決定に必要な論点を明確にした。"
}
```

---

# 評価軸

## 主評価軸

* 論点整理
* 意思決定寄与
* リスク検知
* アクション化

## 補助評価軸

* 根拠性
* 新規性 / 洞察
* 要約・交通整理

## 減点軸

* 重複
* 冗長
* 論点逸脱
* 根拠薄い断言

---

# 発言タイプ

以下のラベルを使います。

* 論点整理
* 提案
* 質問
* 情報共有
* 要約
* 懸念提示
* 根拠提示
* 意思決定促進
* 雑談 / 脱線

---

# スコアリング

主評価軸・補助評価軸は 0〜3 点、減点軸は 0〜-3 点で採点します。

## 総合点

```text
総合点 =
論点整理 × 1.3
+ 意思決定寄与 × 1.5
+ リスク検知 × 1.2
+ アクション化 × 1.3
+ 根拠性 × 0.8
+ 新規性 × 0.9
+ 要約・交通整理 × 0.8
+ 重複
+ 冗長
+ 論点逸脱
+ 根拠薄い断言
```

重みは後で調整できるように、設定ファイルへ切り出す前提です。

---

# 文脈評価

発言単体ではなく、以下を見て評価します。

* 会議目的
* 現在の議題
* 直前3発言
* 対象発言
* 直後3発言

これにより、

* 単なる繰り返しか
* 本当に論点整理だったか
* 発言後に議論が収束したか
  を判断しやすくします。

---

# LLM の役割

LLM は以下を担当します。

1. 発言タイプ分類
2. 軸別採点
3. 減点判定
4. 理由生成

LLM には、発言のうまさではなく、会議目的への寄与で評価させます。

---

# 推奨ディレクトリ構成

```text
project/
  app/
    ingest/
    context_builder/
    evaluators/
    scoring/
    aggregation/
    reporting/
    schemas/
    api/
    ui/
  prompts/
    utterance_eval.txt
  data/
    sample_meetings/
  tests/
  docs/
```

---

# モジュール概要

* `ingest`: 入力正規化
* `context_builder`: 前後文脈生成
* `evaluators`: LLM 評価
* `scoring`: 総合点計算
* `aggregation`: 話者別・会議別集計
* `reporting`: 画面向け整形
* `schemas`: データモデル定義
* `api`: バックエンド API
* `ui`: 可視化 UI

---

# 最初の実装順

1. サンプル会議ログを用意する
2. スキーマを定義する
3. `prompts/utterance_eval.txt` を作る
4. LLM 評価器を実装する
5. 総合点計算を実装する
6. 発言一覧 API を作る
7. 会議サマリー画面を作る
8. 話者別分析画面を作る

---

# セットアップ方針

現時点では実装方針だけを定義し、技術スタックは固定しません。

ただし MVP 実装としては、次の構成を推奨します。

* バックエンド: Python
* スキーマ: Pydantic
* API: FastAPI など
* UI: 軽量な Web UI
* LLM 呼び出し: OpenAI API or モック

---

# OpenAI 設定

実行時は `OPENAI_API_KEY` を設定してください。
`.env` に次のように書いても読み込まれます。

```env
OPENAI_API_KEY=your_api_key_here
```

既定の評価モデルは `gpt-5.4-mini` です。必要なら `config.yaml` の `llm.model` で変更できます。

依存関係の同期は `uv` 前提にしています。

```bash
uv sync
uv run pytest -q
uv run python run.py
```

---

# 判断モデル（採用 LLM）

ローカル推論基盤 (#12) + eval ハーネス (#5) を踏まえて、判断 LLM を Issue #18 で選定しました。

- 第一採用: **`unsloth/Qwen3.6-35B-A3B-NVFP4`** (NVFP4 / compressed-tensors、Blackwell ネイティブ FP4)
- 控え: **`Qwen/Qwen2.5-32B-Instruct-AWQ`** (AWQ Marlin、レイテンシ最速)
- 採用根拠: `docs/adr/0001-judgment-model.md`
- 候補比較: `docs/model_candidates.md` / `docs/model_selection_v1.md`
- 世代管理: `docs/model_history.md`

ステータスは `Proposed`。SSH 先 (RTX 5090 / 32GB) で `scripts/run_model_benchmark.sh --all` を回した実測値で確定 (Accepted) に切り替えます。

## SSH 先での実測手順

```bash
# SSH 先 (RTX 5090) で:
git clone https://github.com/TH-AITeam/meeting-score
cd meeting-score
uv sync
uv pip install vllm

# 全候補を順に回す
bash scripts/run_model_benchmark.sh --all

# 単発（例: 第一採用候補の NVFP4）
MODEL="unsloth/Qwen3.6-35B-A3B-NVFP4" \
SERVED_NAME="qwen3.6-35b-nvfp4" \
EXTRA="--quantization compressed-tensors --enforce-eager" \
bash scripts/run_model_benchmark.sh
```

結果は `reports/model_benchmarks/{served_name}/{ts}_*.json` に出ます。Mac に rsync で持ち帰り、`docs/model_selection_v1.md` の表に転記します。

## 本番設定への切り替え

採用が確定したら `backend/config.yaml.example` を `backend/config.yaml` にコピーし、`llm.endpoint` を実行環境に合わせて書き換えるだけで API が local backend に切り替わります。

---

# 受け入れ条件

以下を満たしたら MVP 完了とみなします。

* 会議文字起こし JSON 1本を読み込める
* 各発言について `speech_type`, `scores`, `penalties`, `total_score`, `reason` を返せる
* 会議を前進させた発言 Top 5 を表示できる
* 話者別サマリーを表示できる
* 重複・冗長・脱線の最低限の検出ができる
* 人が見て明らかに破綻していない

---

# 検証ポイント

* 明らかに良い発言が上位に来るか
* 長く話しただけの人が勝っていないか
* 要約や整理役も評価されるか
* 重複発言が下がるか
* 人が見て納得感があるか

---

# UI の見せ方

以下のような表現を使います。

* 会議を前進させた発言
* 論点を整理した発言
* 意思決定を進めた発言
* リスクを可視化した発言
* 次アクションを明確にした発言

以下は避けます。

* 優秀度ランキング
* 社員ランキング
* 発言力ランキング

---

# 注意事項

このプロジェクトは、**会議参加者の優秀さを断定するシステムではありません**。
会議の振り返りと改善を支援するためのツールとして扱ってください。

この前提を崩す表示や運用は避ける必要があります。

---

# 評価 (eval ハーネス)

`evals/` 配下に、人手アノテと突き合わせて評価器の精度・安定性を測る eval ハーネスを置いています。
詳細スキーマは `data/annotations/README.md` を参照してください。

## 取得メトリクス

* **Spearman / Kendall tau**: 人手ランクとシステムランクの順位相関
* **Top5 Jaccard / Bottom5 Jaccard**: 「貢献した発言」「無価値な発言」の重なり
* **Pairwise accuracy**: ペア比較 (`A_better` / `B_better` / `tie`) の一致率
* **安定性 (Stability)**: 同一発言を N=5 回採点したときの軸別 SD と range

## 実行

```bash
# ベースライン評価
make eval DATASET=data/annotations/gold/v1

# 同一会議を 5 回採点して分散を見る
make eval-stability SAMPLE=data/sample_meetings/sample_meeting_01.json N=5
```

CLI を直接呼びたい場合:

```bash
cd backend && python -m evals.cli run \
  --dataset ../data/annotations/gold/v1 \
  --meetings-dir ../data/annotations/gold/v1/meetings \
  --out ../reports/eval/v1.json
```

## ベースラインスコア（プレースホルダ）

最初のゴールドアノテ (`gold/v1`) が揃い次第ここに数値を載せる。各列は会議横断の macro 平均。

| バージョン | モデル | spearman | kendall_tau | top5_jaccard | bottom5_jaccard | pairwise_acc |
|---|---|---|---|---|---|---|
| v1 | gpt-5.4-mini (デフォルト) | TBD | TBD | TBD | TBD | TBD |

安定性は会議1本につき N=5 で取り、軸別 mean SD / max SD を JSON に書き出す。

---

# 関連ドキュメント

* `AGENT.md`: 実装エージェント向けの作業指示書
* `data/annotations/README.md`: アノテーションスキーマ
* 会議貢献度スコアリング MVP 仕様書: 詳細仕様

---

# 開発手順

## セットアップ

```bash
# Task インストール（初回のみ）
# https://taskfile.dev/installation/
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin

# 依存関係インストール
task setup
```

## よく使うコマンド

```bash
task dev           # バックエンド + フロントエンドを同時起動
task dev:backend   # バックエンドのみ起動（localhost:8000）
task dev:frontend  # フロントエンドのみ起動（localhost:5173）
task lint          # ruff check（構文・スタイル検査）
task format        # ruff format（自動整形）
task format:check  # ruff format --check（整形チェックのみ）
task typecheck     # mypy による型チェック
task test          # pytest でテスト実行
task test:cov      # pytest + カバレッジ（term-missing 表示）
task ci            # CI と同じ lint + test を一括実行
```

タスク一覧は `task --list` で確認できます。

---

# CI (GitHub Actions)

## ワークフロー構成

`main` へのプッシュおよび `main` 向けのプルリクエスト作成時に自動で実行されます。

| ジョブ | 内容 |
|--------|------|
| `frontend` | TypeScript 型チェック・ESLint・Vite ビルド |
| `lint` | `ruff check` / `ruff format --check` / `mypy` |
| `test` | `pytest` + カバレッジ計測（しきい値 70%） |

## ローカルで CI と同じチェックを実行する

```bash
task ci
```

## カバレッジ

- CI でカバレッジが **70% 未満** になるとテストジョブが失敗します
- 安定後は 80% へ引き上げる予定です

## テスト内でのシークレット扱い

`OPENAI_API_KEY` などの外部 API キーを CI に渡さない方針です。LLM を呼び出すテストはモックを使ってください。

---

# CI (GitHub Actions)

## ワークフロー構成

`main` へのプッシュおよび `main` 向けのプルリクエスト作成時に自動で実行されます。

| ジョブ | 内容 |
|--------|------|
| `frontend` | TypeScript 型チェック・ESLint・Vite ビルド |
| `lint` | `ruff check` / `ruff format --check` / `mypy` |
| `test` | `pytest` + カバレッジ計測（しきい値 70%） |

## ローカルで CI と同じチェックを実行する

**フロントエンド**

```bash
cd frontend
npm ci
npm run typecheck
npm run lint
npm run build
```

**バックエンド**

```bash
cd backend

# lint
uv run ruff check .
uv run ruff format --check .
uv run mypy app

# test（カバレッジ付き）
uv run pytest --cov=app --cov-report=term --cov-fail-under=70
```

## カバレッジ

- CI でカバレッジが **70% 未満** になるとテストジョブが失敗します
- 安定後は 80% へ引き上げる予定です
- カバレッジレポートは [Codecov](https://codecov.io/gh/TH-AITeam/meeting-score) で確認できます

## テスト内でのシークレット扱い

`OPENAI_API_KEY` などの外部 API キーを CI に渡さない方針です。LLM を呼び出すテストはモックを使ってください。

---

# 次にやると良いこと

* サンプル会議ログを追加する
* スキーマとモック評価器を実装する
* 最小 API を作る
* UI の初版を作る
* 実データまたは架空データで重みを調整する

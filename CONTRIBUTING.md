# Contributing Guide

## ブランチ命名規則

| プレフィックス | 用途 | 例 |
|---|---|---|
| `feat/` | 新機能 | `feat/utterance-evaluator` |
| `fix/` | バグ修正 | `fix/score-calculation` |
| `chore/` | 設定・依存関係・ドキュメント等 | `chore/update-dependencies` |
| `docs/` | ドキュメントのみ | `docs/api-schema` |
| `test/` | テスト追加・修正 | `test/eval-harness` |
| `refactor/` | 振る舞いを変えないリファクタ | `refactor/scoring-module` |
| `infra/` | CI/CD・インフラ設定 | `infra/github-actions` |
| `decision/` | 技術選定・ADR | `decision/llm-model` |

Issue 番号がある場合は `feat/issue-番号-短い説明` 形式を推奨。

## コミット規約

[Conventional Commits](https://www.conventionalcommits.org/) を推奨。

```
<type>(<scope>): <短い説明>

<本文（任意）>
```

| type | 用途 |
|---|---|
| `feat` | 新機能 |
| `fix` | バグ修正 |
| `chore` | ビルド・設定変更（機能に影響しない） |
| `docs` | ドキュメントのみ |
| `test` | テスト追加・修正 |
| `refactor` | リファクタ（バグ修正でも新機能でもない） |
| `ci` | CI/CD 設定変更 |
| `perf` | パフォーマンス改善 |

スコープ例: `evaluators`, `scoring`, `api`, `ui`, `evals`, `schema`

## PR タイトル形式

コミット規約と同じ形式。

```
feat(evaluators): LLM 評価器の実装
fix(scoring): 重み設定が config.yaml から読まれない問題を修正
```

## 作業フロー

1. `main` から最新を取得してブランチを切る
   ```bash
   git checkout main && git pull
   git checkout -b feat/issue-XX-your-feature
   ```
2. 作業・コミット
3. ローカルで CI 相当を実行して確認
   ```bash
   task ci
   ```
4. PR を作成（`main` ベース）
5. CI が通ることを確認
6. セルフレビュー後にマージ

## PR セルフレビューチェックリスト

PR 本文に以下を含めることを推奨:

- [ ] 変更内容の概要
- [ ] テスト方法（手動確認した内容）
- [ ] 影響範囲（変更が他の機能に与える影響）
- [ ] 残課題・将来対応（あれば）

## マージ戦略

- **Squash merge**: 推奨（デフォルト）。PR 単位でコミット履歴をまとめる
- **Rebase merge**: 直線的な履歴が必要な場合
- **Merge commit**: 禁止

## 緊急ホットフィックス手順

通常は PR フロー必須だが、緊急時は以下の手順でバイパスできる:

1. GitHub の `Settings → Branches` でブランチ保護の **Bypass** を一時的に自分に付与
2. 修正を直接 `main` に push またはブランチなしで対応
3. 対応後すぐにバイパスを外す
4. 事後に PR を作って変更内容を記録する

この手順は **障害対応・本番緊急修正のみ** を想定。通常の開発では使わない。

## ローカルで CI チェックを実行する

```bash
task ci
```

個別に実行する場合:

```bash
task lint          # ruff check
task format:check  # ruff format --check
task typecheck     # mypy
task test          # pytest
```

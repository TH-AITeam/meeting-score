# #22 [infra] pre-commit hooks と補助インフラ

**Labels**: `infra`, `quality`, `P1`
**Milestone**: v0.2

## 概要

ローカルでコミット前に Ruff を走らせて、CI で落ちる前に気付く。 + dependabot, PR template, Issue template, CODEOWNERS を整備。

## 前提

#19 (Ruff 設定済み) が先。

## やること

### pre-commit hooks

- [ ] `.pre-commit-config.yaml` 作成
- [ ] `uv add --dev pre-commit`
- [ ] README に `uv run pre-commit install` 手順追加

### Dependabot

- [ ] `.github/dependabot.yml` 作成
  - Python deps を週1回チェック
  - GitHub Actions も週1回チェック

### PR template

- [ ] `.github/pull_request_template.md` 作成
  - チェックリスト: テスト追加 / ドキュメント更新 / 破壊的変更の明記 / 関連 Issue

### Issue template

- [ ] `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] `.github/ISSUE_TEMPLATE/config.yml` (blank issue 無効化)

### CODEOWNERS

- [ ] `.github/CODEOWNERS` 作成
  - 個人プロジェクトなら全部 `@wyvern623` で OK
  - 将来チームに渡す前提でディレクトリ別オーナーの枠を空けておく

## ファイル例

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: detect-private-key
```

### `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### `.github/pull_request_template.md`

```markdown
## 概要

(何をなぜ変えたか、1-3文)

## 関連 Issue

Closes #

## 変更内容

- [ ] 機能変更
- [ ] バグ修正
- [ ] リファクタリング
- [ ] ドキュメント更新

## チェックリスト

- [ ] テストを追加 or 更新した
- [ ] ローカルで lint / test を通した
- [ ] 破壊的変更がある場合は README / CHANGELOG を更新した
- [ ] 関連 Issue にリンクした

## レビュー観点(セルフレビュー欄)

(レビュアーに見てほしい点)
```

### `.github/ISSUE_TEMPLATE/config.yml`

```yaml
blank_issues_enabled: false
contact_links:
  - name: 質問・議論
    url: https://github.com/wyvern623/meeting-score/discussions
    about: 質問はこちらへ
```

## 完了条件

- `git commit` 時に Ruff が自動実行される
- dependabot が初回 PR を作ってくる
- 新規 PR で template が自動的に挿入される
- 新規 Issue 作成画面でテンプレが選べる

## 注意

- pre-commit を入れた直後の **初回コミット**は全ファイル整形で大変なことになるので、整形コミットを単独で切ること(#19 で一度整形してれば問題なし)
- dependabot の PR が増えすぎたら `open-pull-requests-limit` を絞る
- Issue template と前回作った 17 本の Issue 形式は別物(template は今後の Issue 用)

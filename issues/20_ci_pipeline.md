# #20 [infra] GitHub Actions で CI パイプライン構築

**Labels**: `infra`, `quality`, `P0`
**Milestone**: v0.2

## 概要

PR 作成時に lint + test を自動実行して、人手レビューより前に機械チェックを通す。

## 前提

#19 (Ruff + mypy + pytest-cov 導入) が先。

## やること

- [ ] `.github/workflows/ci.yml` 作成
- [ ] uv のキャッシュ有効化(`astral-sh/setup-uv@v3`)
- [ ] Job 1: lint (ruff check + ruff format --check + mypy)
- [ ] Job 2: test (pytest + coverage、しきい値 70% から開始)
- [ ] Python matrix: 3.13 (将来 3.12 追加検討)
- [ ] PR と main への push の両方で発火
- [ ] バッジを README に追加

## ファイル例 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --frozen
      - name: Ruff check
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check .
      - name: mypy
        run: uv run mypy app

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync --frozen
      - name: pytest
        run: uv run pytest --cov=app --cov-report=xml --cov-report=term --cov-fail-under=70
      - uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: false
        if: always()
```

## 完了条件

- main へ push、 PR 作成で CI が走る
- lint / test の両方がグリーンの状態を維持
- カバレッジしきい値が CI で enforce される
- README に CI バッジ表示

## 将来の追加候補(別 Issue 化してよい)

- 大学 GPU での学習用 self-hosted runner (#13 以降)
- nightly run (時間かかる E2E)
- リリース用 workflow (タグ push で changelog 生成等)

## 注意

- `--frozen` を必ず付ける(uv.lock を尊重)
- 機密値が必要なテストは secrets で渡す(`OPENAI_API_KEY` 等は LLM 評価のテストでは **モック** にして CI に渡さない方針)
- カバレッジしきい値は最初 70%、安定したら 80% へ

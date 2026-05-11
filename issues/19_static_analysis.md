# #19 [infra] 静的解析の導入 (Ruff + mypy)

**Labels**: `infra`, `quality`, `P0`
**Milestone**: v0.2

## 概要

コードフォーマットと lint を Ruff に統一。型チェックは mypy を段階導入。

## 方針

- **Ruff**: lint + formatter (black/flake8/isort 不要)
- **mypy**: `strict = false` でゆるく開始、warn を見ながら徐々に厳しく
- 設定は `pyproject.toml` に集約

## やること

- [ ] `pyproject.toml` に Ruff 設定追加
- [ ] `pyproject.toml` に mypy 設定追加
- [ ] dev dependency 追加: `ruff`, `mypy`, `pytest-cov`
- [ ] `uv run ruff format .` で全コード整形
- [ ] `uv run ruff check . --fix` で自動修正可能なものを直す
- [ ] 残った警告を Issue として個別に切るか、その場で直すかを判断
- [ ] `Makefile` or `tasks.py` に `make lint` / `make format` / `make typecheck` 追加
- [ ] README に開発手順セクション追加

## 推奨設定 (`pyproject.toml` 追記分)

```toml
[tool.ruff]
line-length = 100
target-version = "py313"
exclude = ["data", "checkpoints", ".venv"]

[tool.ruff.lint]
select = [
  "E",    # pycodestyle errors
  "W",    # pycodestyle warnings
  "F",    # pyflakes
  "I",    # isort
  "B",    # flake8-bugbear
  "UP",   # pyupgrade
  "SIM",  # flake8-simplify
  "RUF",  # ruff-specific
  "N",    # pep8-naming
]
ignore = [
  "E501",  # line too long (formatter が処理)
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["N802", "N803"]  # テストは関数名の慣習が違う

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false   # 段階的に true
disallow_incomplete_defs = false
check_untyped_defs = true
no_implicit_optional = true
strict_equality = true
exclude = ["data/", "checkpoints/"]

[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false

[dependency-groups]
dev = [
  "pytest>=8.0.0",
  "pytest-cov>=5.0.0",
  "ruff>=0.6.0",
  "mypy>=1.11.0",
]
```

## 完了条件

- `uv run ruff check .` が exit 0
- `uv run ruff format --check .` が exit 0
- `uv run mypy app` が exit 0(warn のみ)
- README に `make lint` / `make test` の使い方が書かれている

## 注意

- 既存コードのフォーマット変更は **1コミットに分ける**(レビュー困難になるので)
- mypy は完璧にしようとすると無限に時間が溶ける。**最低限のシグネチャ付与で止める**

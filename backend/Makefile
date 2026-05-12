.PHONY: lint format typecheck test

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app

test:
	uv run pytest

test-cov:
	uv run pytest --cov=app --cov-report=term-missing

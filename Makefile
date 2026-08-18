.PHONY: test lint types check fetch

# Dataset adapters live in hyphenated directories and all share the module name
# `adapter`, so mypy cannot walk them together (see the exclude in pyproject.toml).
# They are checked one at a time here so they stay checked rather than skipped.
ADAPTERS := $(wildcard datasets/*/adapter.py) $(wildcard tests/datasets/fixtures/*/adapter.py)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy .
	@for adapter in $(ADAPTERS); do \
		echo "mypy $$adapter"; \
		uv run mypy "$$adapter" || exit 1; \
	done

check: lint types test

fetch:
	uv run python -m scripts.fetch_datasets

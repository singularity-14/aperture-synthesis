.PHONY: install test demo lint clean all

all: install test demo

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

demo:
	python scripts/run_demo.py

lint:
	python -m ruff check src tests scripts

clean:
	rm -rf figures results .pytest_cache src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

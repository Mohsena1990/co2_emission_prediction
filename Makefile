.PHONY: help install test test-leakage fetch-data pipeline tables figures clean

RUN_ID ?= my_run
CONFIG ?= configs/default_config.yaml
SWEEP_CONFIG ?= configs/sweep_config.yaml

help:
	@echo "Q-DECEM - common commands (see README.md for full detail):"
	@echo "  make install       - pip install pinned requirements.txt"
	@echo "  make test          - run the full pytest suite"
	@echo "  make test-leakage  - run only the leakage/nested-CV isolation tests"
	@echo "  make fetch-data    - refresh mobility/grid/OWID caches + build A1-A4 matrices (RUN_ID=$(RUN_ID))"
	@echo "  make pipeline      - run scripts/run_full_pipeline.sh end-to-end (RUN_ID=$(RUN_ID)) - multi-hour"
	@echo "  make tables        - regenerate outputs/tables/ from an existing run (RUN_ID=$(RUN_ID))"
	@echo "  make figures       - regenerate outputs/figures/ from an existing run (RUN_ID=$(RUN_ID))"
	@echo "  make clean         - remove __pycache__ and .pytest_cache directories"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-leakage:
	pytest tests/test_leakage_sentinel.py tests/test_no_leakage.py \
	       tests/test_nested_cv_isolation.py tests/test_pso_isolation.py \
	       tests/test_feature_leakage_fixes.py -v

fetch-data:
	python scripts/fetch_mobility_data.py --config $(CONFIG)
	python scripts/fetch_grid_data.py --config $(CONFIG)
	python scripts/fetch_owid_data.py --config $(CONFIG)
	python scripts/00_make_dataset.py --config $(CONFIG) --run-id $(RUN_ID)

pipeline:
	scripts/run_full_pipeline.sh $(RUN_ID) $(CONFIG) $(SWEEP_CONFIG)

tables:
	python scripts/13_generate_tables.py --config $(SWEEP_CONFIG) --run-id $(RUN_ID)

figures:
	python scripts/14_generate_figures.py --config $(SWEEP_CONFIG) --run-id $(RUN_ID)

clean:
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

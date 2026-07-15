PYTHON := conda run -n eda python
ENV := PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-vnindex

.PHONY: install test lint quick full forecast report clean
install:
	$(PYTHON) -m pip install -e .
test:
	$(ENV) $(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests scripts
quick:
	$(ENV) $(PYTHON) -m vnindex_model.cli run-all --config configs/quick.yaml
full:
	$(ENV) $(PYTHON) -m vnindex_model.cli run-all --config configs/full.yaml
forecast:
	$(ENV) $(PYTHON) -m vnindex_model.cli forecast --config configs/default.yaml
report:
	$(ENV) $(PYTHON) -m vnindex_model.cli report --config configs/default.yaml
clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info


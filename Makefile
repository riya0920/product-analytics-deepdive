.PHONY: generate build test analysis prove all
generate:
	PYTHONPATH=src python -m analytics.generate --users 60000 --days 120
build:
	PYTHONPATH=src python -m analytics.pipeline build
test:
	PYTHONPATH=src python -m analytics.pipeline test
	pytest
analysis:
	PYTHONPATH=src python -m analytics.analysis
prove:
	-PYTHONPATH=src python -m analytics.pipeline test --skip-cleaning
all: generate build test analysis

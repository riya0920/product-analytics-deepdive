.PHONY: generate build test analysis prove charts dbt dbt-docs all
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
charts:
	PYTHONPATH=src python -m analytics.charts
dbt:
	cd dbt && DBT_PROFILES_DIR=. python -m dbt.cli.main build
dbt-docs:
	cd dbt && DBT_PROFILES_DIR=. python -m dbt.cli.main docs generate
all: generate build test analysis charts
query-perf:
	PYTHONPATH=src python -m analytics.query_perf --repeats 5

.PHONY: generate build test analysis prove charts dbt dbt-docs all segments incrementality geo-design
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
segments:
	PYTHONPATH=src python -m analytics.segments
incrementality:
	PYTHONPATH=src python -m analytics.incrementality validate
geo-design:
	PYTHONPATH=src python -m analytics.incrementality design

slice-check:
	PYTHONPATH=src python -m analytics.slice_check
	PYTHONPATH=src python -m analytics.slice_check --by region --focus apac --baseline emea --dims platform channel

lalonde-data:
	PYTHONPATH=src python -m analytics.lalonde_lab --download

lalonde: lalonde-data
	PYTHONPATH=src python -m analytics.lalonde_lab

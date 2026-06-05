.PHONY: install data pipeline app test clean

install:
	pip install -r requirements.txt

data:
	python -m src.data_generation

pipeline:
	python -m src.pipeline

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

clean:
	rm -f data/*.parquet data/*.csv reports/*.json reports/*.csv

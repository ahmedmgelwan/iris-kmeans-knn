data_transformations:
	@echo "Building data transformations..."
	python -m src.pipelines.data_transformation

model_build: data_transformations
	@echo "Building model..."
	python -m src.pipelines.model_training

run_api:
	@echo "Running FastAPI server..."
	uvicorn src.api.main:app --reload
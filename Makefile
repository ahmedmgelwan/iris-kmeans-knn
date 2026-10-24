data_transformations:
	@echo "Building data transformations..."
	python -m src.pipelines.data_transformation

model_build: data_transformations
	@echo "Building model..."
	python -m src.model.model
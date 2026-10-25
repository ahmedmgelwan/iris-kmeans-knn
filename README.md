# Iris Classification API

This project trains a K-Nearest Neighbors (KNN) model on the Iris dataset and
serves predictions through a FastAPI application.

## Project structure

- `src/data/`: data loading, preprocessing, and transformation
- `src/model/`: model training and hyperparameter tuning
- `src/pipelines/`: data transformation and model training pipelines
- `src/api/`: FastAPI application and request/response schemas
- `tests/`: project tests
- `data/`: dataset and generated transformation files
- `assets/`: trained model and transformation pipelines

## Requirements

- Python 3.14
- `uv`

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Setup

Install the project dependencies:

```bash
uv sync
```

## Run the tests

```bash
uv run pytest
```

## Build the data transformations and model

Run the data transformation pipeline:

```bash
make data_transformations
```

Build the model after transforming the data:

```bash
make model_build
```

## Run the API

```bash
make run_api
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction

```http
POST /predict
```

Request body:

```json
{
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response contains the predicted Iris class:

```json
{
  "class_label": "Iris-setosa"
}
```

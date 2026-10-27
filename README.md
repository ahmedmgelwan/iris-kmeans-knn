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

### Track Optuna tuning with MLflow

The tuning run creates an MLflow parent run for the Optuna study and one
nested child run per trial. Each child run records the trial's sampled
hyperparameters and mean cross-validation score. The parent run records the
best parameters, best score, and the winning trial's MLflow run ID.
When `save_best_model` is called, it also logs the fitted estimator as an
MLflow model and registers it under the model name
`IRISClassificationModel`.

By default, runs are stored in the local `mlflow.db` SQLite database. Set
`MLFLOW_TRACKING_URI` to use another MLflow tracking server: MLFLOW_TRACKING_URI=http://127.0.0.1:5000 

```bash
uv run python -m src.pipelines.model_training
```

To inspect the local runs, start the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
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

import joblib
from fastapi import FastAPI

from src.config import config

from .schema import IRISFeatures, IRISTarget

app = FastAPI()
model = joblib.load(config.MODEL_PATH)
feature_pipeline = joblib.load(config.FEATURE_PIPELINE_PATH)
target_pipeline = joblib.load(config.TARGET_PIPELINE_PATH)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: IRISFeatures) -> IRISTarget:
    feature_data = [[features.sepal_width, features.petal_length, features.petal_width]]
    transformed_features = feature_pipeline.transform(feature_data)
    prediction = model.predict(transformed_features)
    return {"class_label": target_pipeline.inverse_transform(prediction)[0]}

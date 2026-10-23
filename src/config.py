import os
from dataclasses import dataclass

@dataclass
class Config:
    RAW_DATASET_PATH: str = os.path.join(os.getcwd(), "data", "iris.csv")
    PREPROCESSED_DATASET_PATH: str = os.path.join(os.getcwd(), "data", "preprocessed_iris.csv")
    FEATURES_PATH: str = os.path.join(os.getcwd(), "data", "features.parquet")
    TARGET_PATH: str = os.path.join(os.getcwd(), "data", "target.parquet")
    ASSETS_PATH: str = os.path.join(os.getcwd(), "assets")
    FEATURE_PIPELINE_PATH: str = os.path.join(os.getcwd(), "assets", "feature_pipeline.pkl")
    TARGET_PIPELINE_PATH: str = os.path.join(os.getcwd(), "assets", "target_pipeline.pkl")
    MODEL_PATH: str = os.path.join(os.getcwd(), "models", "iris_model.pkl")
    RANDOM_STATE: int = 42


config = Config()
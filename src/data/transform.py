import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import setup_logger

logger = setup_logger(__name__)

def transform_features(X, numerical_features) -> tuple:
    """
    Transforms the features of the dataset by applying:
    - simple imputation for missing values
    - standard scaling for numerical features
    """
    transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    X_transformed = transformer.fit_transform(X[numerical_features])
    logger.info("Feature transformation completed successfully.")
    return X_transformed, transformer

def transform_target(y) -> tuple:
    """
    Transforms the target variable by applying label encoding.
    """
    label_encoder = LabelEncoder()
    y_transformed = label_encoder.fit_transform(y)
    logger.info("Target variable transformation completed successfully.")

    return y_transformed, label_encoder


def make_transfomation(X, y, numerical_features) -> tuple:
    """
    Transforms both features and target variable by applying:
    - simple imputation for missing values
    - standard scaling for numerical features
    - label encoding for target variable
    """
    X_transformed, feature_pipeline = transform_features(X, numerical_features)
    y_transformed, target_pipeline = transform_target(y)
    logger.info("Data transformation completed successfully.")
    return X_transformed, y_transformed, feature_pipeline, target_pipeline






def save_transformations(X, y, features_transformed_path, target_transformed_path, feature_pipeline, target_pipeline, feature_pipeline_path, target_pipeline_path) -> bool:
    """
    Saves the transformed data and transformation pipelines for features and target variable to disk.
    """
    joblib.dump(X, features_transformed_path)
    joblib.dump(y, target_transformed_path)
    joblib.dump(feature_pipeline, feature_pipeline_path)
    joblib.dump(target_pipeline, target_pipeline_path)

    logger.info(f"Transformed data and pipelines saved successfully to {feature_pipeline_path} and {target_pipeline_path}.")

    return True

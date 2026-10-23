from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
from src.logger import setup_logger


logger = setup_logger(__name__)

def transform_features(X, numerical_features) -> tuple:
    """
    Transforms the features of the dataset by applying:
    - simple imputation for missing values
    - standard scaling for numerical features
    """

    transformer = ColumnTransformer(
        transformers=[
            ('imputer', SimpleImputer(), numerical_features),
            ('scaler', StandardScaler(), numerical_features)
        ]
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





def save_transformation_pipelines(X, y, feature_pipeline, target_pipeline, feature_pipeline_path, target_pipeline_path) -> None:
    """
    Saves the transformed data and transformation pipelines for features and target variable to disk.
    """

    joblib.dump(feature_pipeline, feature_pipeline_path)
    joblib.dump(target_pipeline, target_pipeline_path)

    logger.info(f"Transformed data and pipelines saved successfully to {feature_pipeline_path} and {target_pipeline_path}.")

    return True

if __name__ == "__main__":
    from src.config import config
    import pandas as pd

    data = pd.read_csv(config.PREPROCESSED_DATASET_PATH)
    X = data.drop(columns=['class'])
    y = data['class']
    numerical_features = X.select_dtypes(include='number').columns.tolist()

    X_transformed, y_transformed, feature_pipeline, target_pipeline = make_transfomation(X, y, numerical_features)
    save_transformation_pipelines(X_transformed, y_transformed, feature_pipeline, target_pipeline, config.FEATURE_PIPELINE_PATH, config.TARGET_PIPELINE_PATH)
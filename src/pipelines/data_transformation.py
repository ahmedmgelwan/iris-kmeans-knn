from src.config import config
from src.data.ingest import load_data
from src.data.preprocess import preprocess_data, save_preprocessed_data
from src.data.transform import make_transfomation, save_transformations


def transform_data():
    data = load_data(config.RAW_DATASET_PATH)
    processed_data = preprocess_data(data)
    save_preprocessed_data(processed_data, config.PREPROCESSED_DATASET_PATH)
    target_column = config.TARGET_COLUMN
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]
    numerical_features = X.select_dtypes(include='number').columns.tolist()

    X_transformed, y_transformed, feature_pipeline, target_pipeline = make_transfomation(X, y, numerical_features)
    save_transformations(X_transformed, y_transformed, 
                        config.FEATURES_TRANSFORMED_PATH,
                        config.TARGET_TRANSFORMED_PATH,
                        feature_pipeline, target_pipeline, 
                        config.FEATURE_PIPELINE_PATH,
                        config.TARGET_PIPELINE_PATH)

transform_data()
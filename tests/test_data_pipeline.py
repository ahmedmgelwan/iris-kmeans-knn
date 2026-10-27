import pandas as pd

from src.config import config
from src.data import ingest, preprocess, transform


def test_ingest_data():
    data = ingest.load_data(config.RAW_DATASET_PATH)
    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'}

def test_preprocess_data():
    data = ingest.load_data(config.RAW_DATASET_PATH)
    preprocessed_data = preprocess.preprocess_data(data)
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert 'sepal_length' not in preprocessed_data.columns
    assert preprocessed_data.isnull().sum().sum() == 0

def test_transform_features():
    data = ingest.load_data(config.PREPROCESSED_DATASET_PATH)
    X = data.drop(columns=['class'])
    numerical_features = X.select_dtypes(include='number').columns.tolist()
    
    y = data['class']
    X_transformed, feature_pipeline = transform.transform_features(X, numerical_features)
    assert X_transformed.shape[0] == X.shape[0]
    assert hasattr(feature_pipeline, 'transform')

def test_transform_target():
    data = ingest.load_data(config.PREPROCESSED_DATASET_PATH)
    X = data.drop(columns=['class'])
    y = data['class']
    y_transformed, target_pipeline = transform.transform_target(y)
    assert y_transformed.shape[0] == y.shape[0]
    assert hasattr(target_pipeline, 'transform')
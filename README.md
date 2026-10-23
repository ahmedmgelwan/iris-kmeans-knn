# End to End IRIS Classification
In this branch I practice basics of `Machine Learning Operation` and build my skills on it.

## Project Structure
- src:
    - data
- data
- tests
- assets

## Using the project
```python
from src.config import config
import pandas as pd

data = pd.read_csv(config.PREPROCESSED_DATASET_PATH)
X = data.drop(columns=['class'])
y = data['class']
numerical_features = X.select_dtypes(include='number').columns.tolist()

X_transformed, y_transformed, feature_pipeline, target_pipeline = make_transfomation(X, y, numerical_features)
print(X_transformed)
save_transformation_pipelines(X_transformed, y_transformed, feature_pipeline, target_pipeline, config.FEATURE_PIPELINE_PATH, config.TARGET_PIPELINE_PATH)
```
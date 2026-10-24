from src.model.model import IRISClassificationModel
from sklearn.neighbors import KNeighborsClassifier

from src.config import config


def train_model():
    params = {
        'n_neighbors': ('int', 1, 20),
        'weights': ('categorical', ['uniform', 'distance']),
        'algorithm': ('categorical', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': ('int', 10, 50),
        'p': ('int', 1, 5)
    }

    model = IRISClassificationModel(KNeighborsClassifier, params)
    model.fit(config.FEATURES_TRANSFORMED_PATH, config.TARGET_TRANSFORMED_PATH)
    model.tune(config.FEATURES_TRANSFORMED_PATH, config.TARGET_TRANSFORMED_PATH, n_trails=10)
    model.save_best_model(config.MODEL_PATH)
    return model

train_model()
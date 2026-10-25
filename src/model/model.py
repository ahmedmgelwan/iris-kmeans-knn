from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import joblib
from src.config import config
from sklearn.model_selection import cross_val_score
import optuna
import pandas as pd

from src.logger import setup_logger


logger = setup_logger(__name__)

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the model with default parameters to the  data and get the mean cross-validation score."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted model."""
        pass

    @abstractmethod
    def tune(self, X, y):

        pass


class IRISClassificationModel(Model):
    def __init__(self, model, params, feature_pipeline_path=config.FEATURE_PIPELINE_PATH, target_pipeline_path=config.TARGET_PIPELINE_PATH):
        self.model_class = model
        self.model = model()
        self.params = params
        self.best_params = None
        self.best_model = None
        self.feature_pipeline = joblib.load(feature_pipeline_path)
        self.target_pipeline = joblib.load(target_pipeline_path) 

    def fit(self, X, y):
        if not hasattr(self.model, 'fit'):
            raise ValueError("The provided model does not have a 'fit' method.")
        if X.endswith('.pkl'):
            X = joblib.load(X)
        if y.endswith('.pkl'):
            y = joblib.load(y)
        self.model.fit(X, y)
        mean_score = cross_val_score(self.model, X, y, cv=5, scoring=config.MODEL_MAIN_METRIC).mean()
        logger.info(f"Cross-validation mean score: {mean_score}")

        return self.model
    

    def predict(self, X):
        if not self.best_model and not self.model:
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method first.")
        X_transformed = self.feature_pipeline.transform(X)
        if  self.best_model:
            return self.target_pipeline.inverse_transform(self.best_model.predict(X_transformed))
        return self.target_pipeline.inverse_transform(self.model.predict(X_transformed))
    

    def tune(self, X, y, n_trails=10):
        if X.endswith('.pkl'):
            X = joblib.load(X)
        if y.endswith('.pkl'):
            y = joblib.load(y)

        def objective(trail):
            params = dict()
            for param_name, pram_config in self.params.items():
                pram_type = pram_config[0]

                if pram_type == 'int':
                    params[param_name] = trail.suggest_int(param_name, pram_config[1], pram_config[2])
                elif pram_type == 'float':
                    params[param_name] = trail.suggest_float(param_name, pram_config[1], pram_config[2])
                elif pram_type == 'categorical':
                    params[param_name] = trail.suggest_categorical(param_name, pram_config[1])

            model = self.model_class(**params)
            return cross_val_score(model, X, y, cv=5, scoring=config.MODEL_MAIN_METRIC).mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trails)
        
        self.best_params = study.best_params
        logger.info(f"Best parameters found: {self.best_params}")
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X, y)
        logger.info(f"Best model found: {self.best_model}")
        return self.best_params

    def save_best_model(self, path=config.MODEL_PATH):
        if not self.best_model:
            raise ValueError("No best model found. Please run the 'tune' method first.")
        joblib.dump(self.best_model, path)
        logger.info(f"Best model saved to {path}.")

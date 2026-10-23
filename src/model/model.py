from abc import ABC, abstractmethod
from flask import logging
from sklearn.model_selection import train_test_split
import joblib
from src.config import config
from src.logger import setup_logger


logger = setup_logger(__name__)

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class IRISClassificationModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        if not hasattr(self.model, 'fit'):
            raise ValueError("The provided model does not have a 'fit' method.")
        if X.endswith('.pkl'):
            X = joblib.load(X)
        if y.endswith('.pkl'):
            y = joblib.load(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
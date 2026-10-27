from abc import ABC, abstractmethod

import joblib
import mlflow
import optuna
from sklearn.model_selection import cross_val_score

from src.config import config
from src.logger import setup_logger

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

logger = setup_logger(__name__)


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the model with default parameters to the  data and get the mean cross-validation score."""

    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted model."""

    @abstractmethod
    def tune(self, X, y):

        pass


class IRISClassificationModel(Model):
    def __init__(
        self,
        model,
        params,
        feature_pipeline_path=config.FEATURE_PIPELINE_PATH,
        target_pipeline_path=config.TARGET_PIPELINE_PATH,
        model_name="IRISClassificationModel",
    ):
        self.model_class = model
        self.model = model()
        self.params = params
        self.best_params = None
        self.best_model = None
        self.feature_pipeline = joblib.load(feature_pipeline_path)
        self.target_pipeline = joblib.load(target_pipeline_path)
        self.model_name = model_name

    def fit(self, X, y):
        if not hasattr(self.model, "fit"):
            raise ValueError("The provided model does not have a 'fit' method.")
        if X.endswith(".pkl"):
            X = joblib.load(X)
        if y.endswith(".pkl"):
            y = joblib.load(y)
        mlflow.set_experiment(self.model_name)
        with mlflow.start_run(run_name=f"{self.model_name}_default_fit"):
            self.model.fit(X, y)
            mean_score = cross_val_score(
                self.model, X, y, cv=5, scoring=config.MODEL_MAIN_METRIC
            ).mean()
            logger.info(f"Cross-validation mean score: {mean_score}")
            mlflow.log_params(self.model.get_params())
            mlflow.log_metric(f"{config.MODEL_MAIN_METRIC}_mean_cv_score", mean_score)

        return self.model

    def predict(self, X):
        if not self.best_model and not self.model:
            raise ValueError(
                "The model has not been fitted yet. Please call the 'fit' method first."
            )
        X_transformed = self.feature_pipeline.transform(X)
        if self.best_model:
            return self.target_pipeline.inverse_transform(
                self.best_model.predict(X_transformed)
            )
        return self.target_pipeline.inverse_transform(self.model.predict(X_transformed))

    def tune(self, X, y, n_trails=10):
        if X.endswith(".pkl"):
            X = joblib.load(X)
        if y.endswith(".pkl"):
            y = joblib.load(y)

        study_name = f"{self.model_name}_optuna"

        with mlflow.start_run(run_name=study_name) as parent_run:
            mlflow.set_tags(
                {
                    "tuning.framework": "optuna",
                    "model.name": self.model_name,
                }
            )

            def objective(trial):
                params = dict()
                for param_name, pram_config in self.params.items():
                    pram_type = pram_config[0]

                    if pram_type == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, pram_config[1], pram_config[2]
                        )
                    elif pram_type == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, pram_config[1], pram_config[2]
                        )
                    elif pram_type == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, pram_config[1]
                        )
                model = self.model_class(**params)
                with mlflow.start_run(
                    run_name=f"{self.model_name}_trial_{trial.number}",
                    nested=True,
                ) as trial_run:
                    score = cross_val_score(
                        model,
                        X,
                        y,
                        cv=5,
                        scoring=config.MODEL_MAIN_METRIC,
                    ).mean()
                    mlflow.log_params(params)
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_metric(
                        f"{config.MODEL_MAIN_METRIC}_mean_cv_score",
                        score,
                    )
                    trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                return score

            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
            )
            study.optimize(objective, n_trials=n_trails)

            mlflow.log_params(study.best_params)
            mlflow.log_metric(
                f"best_{config.MODEL_MAIN_METRIC}_mean_cv_score",
                study.best_value,
            )
            mlflow.log_param("best_trial_number", study.best_trial.number)
            mlflow.log_param("n_trials", len(study.trials))
            mlflow.set_tag("optuna.study_name", study.study_name)
            mlflow.set_tag(
                "optuna.best_trial_run_id", study.best_trial.user_attrs["mlflow_run_id"]
            )
            mlflow.set_tag("mlflow.parent_run_id", parent_run.info.run_id)

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

        with mlflow.start_run(run_name=f"{self.model_name}_best_model") as run:
            mlflow.log_params(self.best_model.get_params())
            mlflow.set_tag("model.lifecycle", "best_model")
            mlflow.set_tag("model.local_path", path)
            mlflow.sklearn.log_model(
                self.best_model,
                name="IRISClassificationModel",
                registered_model_name=self.model_name,
                serialization_format="pickle",
            )
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/IRISClassificationModel",
                name=self.model_name,
            )
            logger.info(
                "Best model registered in MLflow as '%s' from run %s.",
                self.model_name,
                run.info.run_id,
            )

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.config import config
from src.model.model import IRISClassificationModel


def make_model() -> IRISClassificationModel:
    return IRISClassificationModel(
        KNeighborsClassifier,
        {"n_neighbors": ("int", 1, 3)},
    )


def test_fit_trains_an_estimator_and_predicts_original_label():
    model = make_model()

    fitted_model = model.fit(
        config.FEATURES_TRANSFORMED_PATH,
        config.TARGET_TRANSFORMED_PATH,
    )
    sample = pd.DataFrame(
        [[3.5, 1.4, 0.2]],
        columns=["sepal_width", "petal_length", "petal_width"],
    )

    assert isinstance(fitted_model, KNeighborsClassifier)
    assert model.predict(sample).tolist() == ["Iris-setosa"]


def test_tune_fits_the_best_model():
    model = make_model()

    best_params = model.tune(
        config.FEATURES_TRANSFORMED_PATH,
        config.TARGET_TRANSFORMED_PATH,
        n_trails=1,
    )

    assert model.best_model is not None
    assert best_params == model.best_params

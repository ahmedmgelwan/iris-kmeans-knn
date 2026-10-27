import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data by:
    - handling missing values
    - droping `sepal_length` column
    """

    data = data.dropna()
    data = data.drop(columns=["sepal_length"])

    logger.info("Data preprocessing completed successfully.")

    return data


def save_preprocessed_data(data: pd.DataFrame, path: str) -> None:
    """Save the preprocessed data to a CSV file."""
    logger.info(f"Saving preprocessed data to {path}.")
    data.to_csv(path, index=False)

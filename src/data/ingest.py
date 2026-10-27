import os

import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""

    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        raise RuntimeError(f"An error occurred while loading the data: {e}")

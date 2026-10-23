import logging

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Function to setup a logger with the specified name, log file, and logging level."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    )

    return logging.getLogger(name)
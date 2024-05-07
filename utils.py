from logging import (
    getLogger,
    Logger,
    Formatter,
    FileHandler,
    StreamHandler,
)
from datetime import datetime
from os import makedirs
from typing import Union


def init_logger(
    name: str = "System", log_level: Union[str, None] = "INFO", save: bool = True
) -> Logger:
    logger = getLogger(name)
    if log_level is None:
        logger.propagate = False
    else:
        logger.setLevel("DEBUG" if log_level is None else log_level.upper())
        formatter = Formatter("[%(asctime)s %(levelname)s - %(name)s] %(message)s")
        formatter.datefmt = "%H:%M:%S"

        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if save:
            makedirs("./logs", exist_ok=True)
            file_handler = FileHandler(
                datetime.now().strftime(f"./logs/{name}_%Y%m%d%H%M%S.log")
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger

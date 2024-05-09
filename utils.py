from logging import (
    getLogger,
    Logger,
    Formatter,
    FileHandler,
    StreamHandler,
)
from datetime import datetime
from os import makedirs
from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
from cv2.typing import MatLike
from yaml import safe_load


def init_logger(
    name: str = "System", log_level: str | None = "INFO", save: bool = True
) -> Logger:
    logger = getLogger(name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    if log_level is None or log_level.upper() == "NONE":
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


def load_configuration(path: str) -> dict:
    return safe_load(open(path, "r"))


def draw_annotation(
    image: MatLike, bboxes: list[tuple[int]], classes: list[str]
) -> MatLike:
    draw = image.copy()
    shape = draw.shape
    for bbox in bboxes:
        c, x, y, w, h = (
            int(bbox[0]),
            int(bbox[1] * shape[1]),
            int(bbox[2] * shape[0]),
            int(bbox[3] * shape[1]),
            int(bbox[4] * shape[0]),
        )
        top_left = (x - w // 2, y - h // 2)
        bottom_right = (x + w // 2, y + h // 2)

        draw = rectangle(draw, top_left, bottom_right, (0, 255, 0), 2)
        label = classes[c]

        draw = putText(
            draw,
            label,
            (top_left[0], top_left[1] - 5),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return draw

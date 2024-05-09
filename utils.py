import logging
import datetime
import os
import cv2
import yaml


def init_logger(
    name: str = "System", log_level: str | None = "INFO", save: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    if log_level is None or log_level.upper() == "NONE":
        logger.propagate = False
    else:
        logger.setLevel("DEBUG" if log_level is None else log_level.upper())
        formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s - %(name)s] %(message)s"
        )
        formatter.datefmt = "%H:%M:%S"

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if save:
            os.makedirs("./logs", exist_ok=True)
            file_handler = logging.FileHandler(
                datetime.datetime.now().strftime(f"./logs/{name}_%Y%m%d%H%M%S.log")
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def load_configuration(path: str) -> dict:
    return yaml.safe_load(open(path, "r"))


def draw_annotation(
    image: cv2.typing.MatLike, bboxes: list[tuple[int]], classes: list[str]
) -> cv2.typing.MatLike:
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

        draw = cv2.rectangle(draw, top_left, bottom_right, (0, 255, 0), 2)
        label = classes[c]

        draw = cv2.putText(
            draw,
            label,
            (top_left[0], top_left[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return draw


def preprocess(images: cv2.typing.MatLike, input_size: int) -> cv2.typing.MatLike:
    processed_images = []
    for image in images:
        processed_image = image.copy()
        processed_image = cv2.resize(processed_image, (input_size, input_size))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess.astype("float32")
        processed_image /= 255.0
        processed_image = processed_image.transpose((2, 0, 1))
        processed_images.append(processed_image)
    return processed_images

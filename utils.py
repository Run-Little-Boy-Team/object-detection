from logging import getLogger, basicConfig, Logger


def init_logger(name: str = "System", log_level: str = "INFO") -> Logger:
    logger = getLogger(name)
    basicConfig(
        format="[%(asctime)s %(levelname)s - %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )
    return logger

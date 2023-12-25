import logging
import logging.handlers

logger = logging.getLogger()


class _LogHandler:
    """
    Define log handler
    """

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    def __init__(self, level: str) -> None:
        self.level = level

    def stream_handler(self) -> logging.Handler:
        """
        Log to console.

        :return: Handler
        """
        handler = logging.StreamHandler()
        handler.setLevel(self.level)
        handler.setFormatter(self.formatter)
        return handler


def init_logger(level: str = "INFO"):
    """
    Initialize logger.

    :return: None
    """
    handler = _LogHandler(level)
    logger.setLevel(handler.level)
    logger.addHandler(handler.stream_handler())

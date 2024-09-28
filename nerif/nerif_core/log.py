import logging
import sys


def set_up_logging(
    out_file: None | str = None, 
    mode: str = "a", 
    fmt: str = "%(levelname)s\t%(name)s\t%(message)s", 
    std: bool = False,
    level: int | str = logging.DEBUG
):

    logger = logging.getLogger("Nerif")
    logger.setLevel(level)

    basic_formatting = logging.Formatter(fmt)

    if out_file != None:
        file_handler = logging.FileHandler(out_file, mode=mode)
        file_handler.setFormatter(basic_formatting)
        logger.addHandler(file_handler)

    if std:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(basic_formatting)
        logger.addHandler(stream_handler)

    if std or out_file != None:
        logger.info("-" * 20)
        logger.info("logging enabled")

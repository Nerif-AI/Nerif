import logging
import sys


def set_up_logging(out="out.log", mode="a", fmt="%(levelname)s\t%(name)s\t%(message)s", std=False):

    logger = logging.getLogger("Nerif")
    logger.setLevel(logging.DEBUG)

    # setting logging format

    basic_formatting = logging.Formatter(fmt)

    # logging into file
    file_handler = logging.FileHandler(out, mode=mode)
    file_handler.setFormatter(basic_formatting)
    logger.addHandler(file_handler)

    if std:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(basic_formatting)
        logger.addHandler(stream_handler)
    
    logger.info("-" * 20)
    logger.info("logging enabled")

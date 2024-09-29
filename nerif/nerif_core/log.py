import logging
import sys
import re
from typing import Literal
# import json
import ast

INDENT = "\t"

def set_up_logging(
    out_file: None | str = None,
    mode: Literal["a", "w"] = "a",
    fmt: str = "%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
    std: bool = False,
    level: int | str = logging.DEBUG,
):

    logger = logging.getLogger("Nerif")
    logger.setLevel(level)

    basic_formatting = NerifFormatter(fmt)

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

# FIXME the name of formatter collide with nerif format, rethink a name
class NerifFormatter(logging.Formatter):
    def __init__(
        self, fmt=None, datefmt=None, style="%", validate=True, *, defaults=None
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

    def format(self, record):

        s = super().format(record)
        r = re.sub(
            r"<dict>(.*)</dict>", 
            lambda x: NerifFormatter.evaller(x),
            s
        )

        return r
    
    @staticmethod
    def evaller(d):
        # FIXME only able to take the first item
        s = ast.literal_eval(d.group(1))
        return NerifFormatter.prettify_dict(s)

    # FIXME the classname is actually not dict, refactor it later
    @staticmethod
    def prettify_dict(d, indented=2):
        if type(d) == dict:
            item = [f"{INDENT * indented}{k}: {NerifFormatter.prettify_dict(v, indented + 1)}" for k,v in d.items()]
            item.append(f"{INDENT * (indented - 1)}}}")
            item = "\n".join(item)
            return f"{{\n{item}"
        if type(d) == list:
            item = [f"{INDENT * indented}{NerifFormatter.prettify_dict(k, indented + 1)}" for k in d]
            item.append(f"{INDENT * (indented - 1)}]")
            item = "\n".join(item)
            return f"[\n{item}"
        return str(d)

import logging
import sys
import re
from typing import Literal
from datetime import datetime
import ast

INDENT = "\t"

def set_up_logging(
    out_file: None | str = None,
    time_stamp: bool = True,
    mode: Literal["a", "w"] = "a",
    fmt: str = "%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
    std: bool = False,
    level: int | str = logging.DEBUG,
):

    logger = logging.getLogger("Nerif")
    logger.setLevel(level)

    basic_formatting = NerifFormatter(fmt)

    if out_file != None:
        if time_stamp:
            p_ext = out_file.rindex('.')
            t_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            out_file = f"{out_file[:p_ext]} {t_string}{out_file[p_ext:]}"

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
        try:
            s = ast.literal_eval(d.group(1))
            return NerifFormatter.prettify(s)
        except ValueError:
            return f"\n{d.group(1)}\n(parsing error happened)\n"


    @staticmethod
    def prettify(d, indented=2):
        rec = NerifFormatter.prettify
        if type(d) == dict:
            item = [
                f"{rec(k)}: {rec(v, indented + 1)},"
                for k,v in d.items()
            ]
            item.insert(0, "{")
            item_string = f"\n{INDENT * indented}".join(item)
            unindented  = INDENT * (indented - 1)
            return "%s\n%s}" % (item_string, unindented)
        
        if type(d) == list:
            item = [f"{rec(k, indented + 1)}," for k in d]
            item.insert(0, "[")
            item_string = f"\n{INDENT * indented}".join(item)
            unindented  = INDENT * (indented - 1)
            return "%s\n%s]" % (item_string, unindented)

        if type(d) == str:
            return f"\"{d}\""
        
        return str(d)

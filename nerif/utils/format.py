import re


class FormatVerifierBase:
    cls = object
    simple = True

    def verify(self, val: str) -> bool:
        raise NotImplementedError("Verify method of FormatVerifierBase is not implemented")

    def match(self, val: str) -> any:
        raise NotImplementedError("Match method of FormatVerifierBase is not implemented")

    def convert(self, val: str) -> any:
        raise NotImplementedError("Convert method of FormatVerifierBase is not implemented")

    def __call__(self, val: str) -> any:
        if self.verify(val):
            return self.convert(val)
        else:
            res = self.match(val)
            if res is not None:
                return res
            else:
                raise ValueError("Cannot convert {} to {}".format(val, self.cls.__name__))


class FormatVerifierInt(FormatVerifierBase):
    cls = int
    pattern = re.compile(r"\b\d+\b")

    # check if the string is a number
    def verify(self, val: str) -> bool:
        return val.isdigit()

    # extract the number from the string
    def match(self, val: str) -> int:
        candidate = self.pattern.findall(val)
        if len(candidate) > 0:
            return int(candidate[0])
        return None

    # type converter
    def convert(self, val: str) -> int:
        return int(val)


class FormatVerifierFloat(FormatVerifierBase):
    cls = float
    pattern = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")

    def verify(self, val: str) -> bool:
        return self.pattern.match(val) is not None

    def match(self, val: str) -> any:
        candidate = self.pattern.findall(val)
        if len(candidate) > 0:
            return float(candidate[0])
        return None

    def convert(self, val: str) -> float:
        return float(val)


class FormatVerifierListInt(FormatVerifierBase):
    cls = list[int]
    simple = False
    pattern = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

    def verify(self, val: str) -> bool:
        have_bound = val.startswith("[") and val.endswith("]")
        if have_bound:
            val = val[1:-1]
            val = val.split(",")
            for v in val:
                if not v.isdigit():
                    return False
            return True
        return False

    def match(self, val: str) -> list[int]:
        candidate = self.pattern.findall(val)
        if len(candidate) > 0:
            return self.convert(candidate[0])
        return None

    def convert(self, val: str) -> list[int]:
        if self.verify(val):
            val = val[1:-1]
            val = val.split(",")
            return [int(v) for v in val]
        return None


class FormatVerifierHumanReadableList(FormatVerifierBase):
    cls = list[str]
    simple = False
    pattern = re.compile(r"^\s*\d+\.\s+.*?(?=\n\s*\d+\.|\Z)", re.MULTILINE | re.DOTALL)

    def verify(self, val: str) -> bool:
        return False  # Force verifier to use match method

    def match(self, val: str) -> list[str]:
        matches = self.pattern.findall(val)
        matches = [match.strip() for match in matches]
        matches = [match.split(". ")[1].strip() for match in matches]
        if len(matches) > 0:
            return matches
        return None

    def convert(self, val: str) -> list[str]:
        return self.match(val)


class FormatVerifierStringList(FormatVerifierBase):
    cls = list[str]
    simple = False
    pattern = re.compile(r'\[(?:\s*(?:"[^"]*"|\"[^\"]*\")\s*,?)*\s*\]', flags=re.DOTALL)

    def verify(self, val: str) -> bool:
        have_bound = val.startswith("[") and val.endswith("]")
        if have_bound:
            val = val[1:-1]
            val = val.split(",")
            return True
        return False

    def match(self, val: str) -> list[int]:
        candidate = self.pattern.findall(val)
        if len(candidate) > 0:
            return self.convert(candidate[0])
        # If regex pattern didn't match, try alternative string list formats:
        # 1. Python list with mismatched quotes
        # 2. Markdown-style bullet points
        # 3. Numbered lists
        if "[" in val and "]" in val:
            candidate = val[val.find("[") : val.rfind("]") + 1]
            candidate = candidate.split(",")
            candidate = [x.strip()[1:-1] for x in candidate]
            return candidate

        # Not a valid Python list - attempt to parse as a Markdown-style list (e.g. "- item" or "* item")
        candidate = val.split("\n")
        candidate = [x.strip() for x in candidate if x.strip() != ""]
        candidate = [x for x in candidate if x.startswith("-") or x.startswith("*") or x.startswith("+")]
        candidate = [x[1:].strip() for x in candidate]
        if len(candidate) > 0:
            return candidate
        # Try parsing as a numbered list format (e.g. "1. item")
        index_pattern = re.compile(r"\d+\.\s")
        candidate = val.split("\n")
        candidate = [x.strip() for x in candidate]
        candidate = [x for x in candidate if index_pattern.match(x)]
        candidate = [x[x.find(".") :].strip() for x in candidate]
        if len(candidate) > 0:
            return candidate
        return None

    def convert(self, val: str) -> list[int]:
        try:
            res = eval(val.strip())
        except Exception as e:
            print("Cannot convet because error {}".format(e))
        return res


class NerifFormat:
    """
    Convert llm response to given type
    """

    def __init__(self):
        pass

    def try_convert(self, val: str, verifier_cls: FormatVerifierBase = None):
        """
        Try to convert the value to the given type
        """
        assert verifier_cls is not None, "Verifier is not given"
        verifier = verifier_cls()
        return verifier(val)

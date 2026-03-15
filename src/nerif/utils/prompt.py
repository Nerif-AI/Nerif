"""Lightweight prompt template with variable substitution."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class PromptTemplate:
    """Prompt template with {variable} substitution, defaults, and conditional sections.

    Variable substitution:
        PromptTemplate("Hello {name}").format(name="World")  # "Hello World"

    Defaults:
        PromptTemplate("In {lang}", defaults={"lang": "English"}).format()  # "In English"

    Conditional sections — gated on the first variable found inside:
        PromptTemplate("Do X.{? format: {fmt}}").format()            # "Do X."
        PromptTemplate("Do X.{? format: {fmt}}").format(fmt="JSON")  # "Do X. format: JSON"

    Conditional sections can contain multiple variables:
        PromptTemplate("{? {a} and {b}}").format(a="X", b="Y")  # " X and Y"
        PromptTemplate("{? {a} and {b}}").format()               # "" (a is trigger, absent)

    Inside conditional sections, variables use silent substitution (empty string
    if missing). Top-level variables raise KeyError if missing.
    """

    _VAR_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(self, template: str, defaults: Optional[Dict[str, Any]] = None):
        self.template = template
        self.defaults = defaults or {}

    @staticmethod
    def _find_conditionals(template: str) -> List[Tuple[int, int, str]]:
        """Find all {? ... } sections, handling nested {var} braces.

        Returns list of (start, end, inner_content) tuples.
        """
        results = []
        i = 0
        while i < len(template) - 1:
            if template[i : i + 2] == "{?":
                # Find the matching closing brace, accounting for nested {var}
                depth = 1
                start = i
                j = i + 2
                while j < len(template) and depth > 0:
                    if template[j] == "{":
                        depth += 1
                    elif template[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    # inner content is between {? and the final }
                    inner = template[start + 2 : j - 1]
                    results.append((start, j, inner))
                    i = j
                    continue
            i += 1
        return results

    @property
    def variables(self) -> Set[str]:
        """Return all variable names in the template (excluding conditional markers)."""
        # Strip conditional markers, keep inner content
        cleaned = self.template
        for start, end, inner in reversed(self._find_conditionals(self.template)):
            cleaned = cleaned[:start] + inner + cleaned[end:]
        return set(self._VAR_PATTERN.findall(cleaned))

    def format(self, **kwargs) -> str:
        """Render the template with the given variables.

        Raises:
            KeyError: If a required top-level variable is missing.
        """
        merged = {**self.defaults, **kwargs}

        # Process conditional sections (in reverse to preserve indices)
        result = self.template
        for start, end, inner in reversed(self._find_conditionals(result)):
            var_names = self._VAR_PATTERN.findall(inner)
            if not var_names:
                # No variables in conditional — include as-is
                result = result[:start] + inner + result[end:]
                continue
            trigger = var_names[0]
            if trigger in merged and merged[trigger] is not None:
                # Render section; all variables use silent substitution
                rendered = self._VAR_PATTERN.sub(lambda m: str(merged.get(m.group(1), "")), inner)
                result = result[:start] + rendered + result[end:]
            else:
                # Remove entire conditional section
                result = result[:start] + result[end:]

        # Substitute remaining top-level variables (strict — raises KeyError)
        def replace_var(match):
            name = match.group(1)
            if name not in merged:
                raise KeyError(f"Missing template variable: {name}")
            return str(merged[name])

        return self._VAR_PATTERN.sub(replace_var, result)

    def partial(self, **kwargs) -> "PromptTemplate":
        """Return a new template with some variables pre-filled as defaults."""
        new_defaults = {**self.defaults, **kwargs}
        return PromptTemplate(self.template, defaults=new_defaults)

    def __repr__(self) -> str:
        return f"PromptTemplate({self.template!r})"

    def __add__(self, other: "PromptTemplate") -> "PromptTemplate":
        """Concatenate two templates."""
        combined = self.template + other.template
        merged_defaults = {**self.defaults, **other.defaults}
        return PromptTemplate(combined, defaults=merged_defaults)

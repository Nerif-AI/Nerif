"""Lightweight prompt template with variable substitution."""

import re
from typing import Any, Dict, Optional, Set


class PromptTemplate:
    """Prompt template with {variable} substitution, defaults, and conditional sections.

    Variable substitution:
        PromptTemplate("Hello {name}").format(name="World")  # "Hello World"

    Defaults:
        PromptTemplate("In {lang}", defaults={"lang": "English"}).format()  # "In English"

    Conditional sections — gated on a single trigger variable:
        PromptTemplate("Do X.{? format: {fmt}}").format()            # "Do X."
        PromptTemplate("Do X.{? format: {fmt}}").format(fmt="JSON")  # "Do X. format: JSON"

    Inside conditional sections, variables use silent substitution (empty string
    if missing). Top-level variables raise KeyError if missing.
    """

    _VAR_PATTERN = re.compile(r"\{(\w+)\}")
    _CONDITIONAL_PATTERN = re.compile(r"\{\?([^}]*\{(\w+)\}[^}]*)\}")

    def __init__(self, template: str, defaults: Optional[Dict[str, Any]] = None):
        self.template = template
        self.defaults = defaults or {}

    @property
    def variables(self) -> Set[str]:
        """Return all variable names in the template (including inside conditionals)."""
        return set(self._VAR_PATTERN.findall(self.template))

    def format(self, **kwargs) -> str:
        """Render the template with the given variables.

        Raises:
            KeyError: If a required top-level variable is missing.
        """
        merged = {**self.defaults, **kwargs}

        def replace_conditional(match):
            section = match.group(1)
            var_name = match.group(2)
            if var_name in merged and merged[var_name] is not None:
                return self._VAR_PATTERN.sub(lambda m: str(merged.get(m.group(1), "")), section)
            return ""

        result = self._CONDITIONAL_PATTERN.sub(replace_conditional, self.template)

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

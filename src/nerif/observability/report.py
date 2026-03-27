"""HTML report generation for traces. Requires nerif[viz] (jinja2)."""
from __future__ import annotations

import os
from typing import Dict

try:
    import jinja2
except ImportError:
    raise ImportError(
        "HTML reports require jinja2. Install with: pip install nerif[viz]"
    )

from .tracing import Trace

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class TraceReport:
    def __init__(self, trace: Trace):
        self.trace = trace

    def generate(self, output_path: str = "trace_report.html") -> str:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
            autoescape=True,
        )
        template = env.get_template("trace_report.html.j2")

        parent_names: Dict[str, str] = {}
        for span in self.trace.spans:
            parent_names[span.span_id] = span.name

        # Build ASCII tree
        lines = []
        by_parent = {}
        for s in self.trace.spans:
            by_parent.setdefault(s.parent_span_id, []).append(s)
        self._render_tree(self.trace.root_span, by_parent, lines, "", True)
        span_tree = "\n".join(lines)

        duration_ms = (self.trace.end_time - self.trace.start_time) * 1000.0

        html = template.render(
            trace=self.trace,
            parent_names=parent_names,
            span_tree=span_tree,
            duration_ms=duration_ms,
        )

        with open(output_path, "w") as f:
            f.write(html)
        return output_path

    def _render_tree(self, span, by_parent, lines, prefix, is_last):
        connector = "└── " if is_last else "├── "
        dur = f"{span.duration_ms():.0f}ms" if span.duration_ms() is not None else "?"
        tokens = f"tokens:{span.token_usage.total_tokens}" if span.token_usage else ""
        lines.append(f"{prefix}{connector}[{span.name}] {dur}  {tokens}".rstrip())
        children = by_parent.get(span.span_id, [])
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            self._render_tree(child, by_parent, lines, child_prefix, i == len(children) - 1)

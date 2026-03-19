"""Shared constants for the Nerif package."""

import httpx

# Default httpx timeout (30 seconds connect, 120 seconds read for LLM responses)
DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)

# Logger hierarchy root name — all nerif loggers use "nerif" or "nerif.*"
LOGGER_NAME = "nerif"

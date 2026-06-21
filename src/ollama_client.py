"""Shared Ollama client helpers with bounded timeouts."""

from __future__ import annotations

import os

import ollama

from src.config import SETTINGS


def get_client(timeout_seconds: int | None = None) -> ollama.Client:
    timeout = int(timeout_seconds or SETTINGS.ollama_timeout_seconds)
    host = os.environ.get("OLLAMA_HOST") or SETTINGS.ollama_host
    if host:
        return ollama.Client(host=host, timeout=timeout)
    return ollama.Client(timeout=timeout)


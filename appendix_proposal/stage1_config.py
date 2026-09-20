"""Small dependency-free config loader for Stage 1 scripts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() in {"null", "none"}:
        return None
    if text.startswith("[") or text.startswith("(") or text.startswith("{"):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            if text.startswith("[") and text.endswith("]"):
                inner = text[1:-1].strip()
                if not inner:
                    return []
                return [_parse_scalar(part) for part in inner.split(",")]
            raise
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text.strip("'\"")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, value = line.split(":", 1)
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent + 2, child))
        else:
            parent[key.strip()] = _parse_scalar(value)
    return root


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return {} if loaded is None else dict(loaded)
    except Exception:
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_simple_yaml(text)

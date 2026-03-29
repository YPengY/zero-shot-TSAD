"""Small JSON and JSONL helpers shared by CLI and workbench code."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def read_json_file(path: str | Path) -> Any:
    """Read one UTF-8 JSON file and return the decoded payload."""

    json_path = Path(path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def read_json_mapping(path: str | Path) -> dict[str, Any]:
    """Read one JSON file and require the top-level payload to be a mapping.

    This is used for configuration files and structured reports where callers
    depend on stable object-style access rather than arbitrary JSON payloads.
    """

    payload = read_json_file(path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Expected a JSON object in `{Path(path)}`, got `{type(payload).__name__}`."
        )
    return dict(payload)


def write_json_file(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
) -> Path:
    """Write one JSON payload with a stable UTF-8 encoding."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_path


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSONL rows and require each row to be a mapping.

    JSONL files in this project are treated as object streams. Raising early on
    scalar or array rows keeps downstream metric/report readers predictable.
    """

    jsonl_path = Path(path)
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        row_text = line.strip()
        if not row_text:
            continue
        row = json.loads(row_text)
        if not isinstance(row, Mapping):
            raise ValueError(
                f"Expected JSON object rows in `{jsonl_path}`:{line_number}, "
                f"got `{type(row).__name__}`."
            )
        yield dict(row)


def read_first_jsonl_mapping(path: str | Path) -> dict[str, Any] | None:
    """Return the first non-empty JSONL row or `None` when the file is empty."""

    return next(iter_jsonl(path), None)

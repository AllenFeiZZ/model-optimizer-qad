#!/usr/bin/env python3
"""Resolve ${NAME} placeholders in a YAML file without changing YAML value types."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve(value):
    if isinstance(value, str):
        missing = sorted({name for name in ENV_PATTERN.findall(value) if name not in os.environ})
        if missing:
            raise ValueError(f"Unset environment variable(s): {', '.join(missing)}")
        return ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    return value


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT_YAML OUTPUT_YAML")

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    config = yaml.safe_load(source.read_text())
    destination.write_text(yaml.safe_dump(resolve(config), sort_keys=False))


if __name__ == "__main__":
    main()

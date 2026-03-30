#!/usr/bin/env python3
"""Fetch the latest snowflake-ml-python version from PyPI and prepend it to a base requirements file."""

import json
import re
import sys
import urllib.request
from typing import Optional


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a PEP 440 version string into a tuple of ints for sorting."""
    return tuple(int(x) for x in re.findall(r"\d+", v))


def get_latest_pypi_version(package_name: str) -> Optional[str]:
    """Get the latest non-yanked version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

        # PyPI "info.version" is already the latest stable release
        latest: str = data["info"]["version"]
        releases = data["releases"].get(latest, [])
        if releases and not releases[0].get("yanked", False):
            return latest

        # Fall back: scan all releases for the latest non-yanked version
        versions = sorted(data["releases"].keys(), key=_parse_version, reverse=True)
        for v in versions:
            files = data["releases"][v]
            if files and not files[0].get("yanked", False):
                return str(v)
    return None


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: update_model_logger_version.py <base-requirements-file>")

    base_file = sys.argv[1]
    latest_version = get_latest_pypi_version("snowflake-ml-python")

    if latest_version is None:
        raise ValueError("Could not get latest version of snowflake-ml-python")

    with open(base_file) as f:
        base_content = f.read()

    sys.stdout.write(f"snowflake-ml-python=={latest_version}\n")
    sys.stdout.write(base_content)


if __name__ == "__main__":
    main()

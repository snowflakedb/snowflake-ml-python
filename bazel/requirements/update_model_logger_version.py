#!/usr/bin/env python3
"""Fetch the latest snowflake-ml-python version from PyPI and prepend it to a base requirements file."""

import json
import sys
import urllib.request
from typing import Optional

from packaging import version


def get_latest_pypi_version(package_name: str) -> Optional[str]:
    """Get the latest non-yanked version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

        versions = data["releases"].keys()
        versions = [version.parse(v) for v in versions]
        versions.sort(reverse=True)

        # find the last version that's not yanked
        for v in versions:
            if not data["releases"][str(v)][0]["yanked"]:
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

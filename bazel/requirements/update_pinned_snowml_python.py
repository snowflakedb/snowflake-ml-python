#!/usr/bin/env python3
"""Generate pinned snowflake-ml-python version files from PyPI for container images."""

import argparse
import json
import re
import sys
import urllib.request
from typing import Optional

SNOWML_PYTHON_PIN_PLACEHOLDER = "__SNOWML_PYTHON_PIN__"


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a PEP 440 version string into a tuple of ints for sorting."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


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
        for version in versions:
            files = data["releases"][version]
            if files and not files[0].get("yanked", False):
                return str(version)
    return None


def _replace_pin_placeholder(base_content: str, pin_line: str) -> str:
    """Replace the shared snowflake-ml-python pin placeholder in a base file."""
    if SNOWML_PYTHON_PIN_PLACEHOLDER not in base_content:
        raise ValueError(f"Missing {SNOWML_PYTHON_PIN_PLACEHOLDER} placeholder in base file.")
    return base_content.replace(SNOWML_PYTHON_PIN_PLACEHOLDER, pin_line, 1)


def generate_pip_requirements(base_content: str, latest_version: str) -> str:
    """Replace the pip placeholder with a snowflake-ml-python pin."""
    return _replace_pin_placeholder(base_content, f"snowflake-ml-python=={latest_version}")


def generate_conda_env(base_content: str, latest_version: str) -> str:
    """Replace the placeholder with a pip-style snowflake-ml-python pin in conda env files."""
    return _replace_pin_placeholder(base_content, f"snowflake-ml-python=={latest_version}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate container dependency files with the latest snowflake-ml-python from PyPI.",
    )
    parser.add_argument("base_file", help="Base requirements or conda file without the snowflake-ml-python pin.")
    parser.add_argument(
        "--format",
        choices=["pip", "conda"],
        default="pip",
        help="Output format: pip requirements.txt prefix (default) or conda environment pin.",
    )
    args = parser.parse_args()

    latest_version = get_latest_pypi_version("snowflake-ml-python")
    if latest_version is None:
        raise ValueError("Could not get latest version of snowflake-ml-python")

    with open(args.base_file) as file:
        base_content = file.read()

    if args.format == "pip":
        output = generate_pip_requirements(base_content, latest_version)
    else:
        output = generate_conda_env(base_content, latest_version)

    sys.stdout.write(output)


if __name__ == "__main__":
    main()

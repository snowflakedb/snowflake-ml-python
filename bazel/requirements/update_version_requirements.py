#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import Optional

import requests
from packaging.requirements import Requirement
from packaging.version import Version, parse as parse_version
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True  # type: ignore[assignment]
yaml.explicit_start = True  # type: ignore[assignment]
yaml.indent(mapping=2, sequence=2, offset=0)

DEFAULT_CONDA_CHANNEL = "https://repo.anaconda.com/pkgs/snowflake"
PYPI_BASE_URL = "https://pypi.org/pypi"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_latest_pypi_version(pkg_name: str) -> Optional[Version]:
    """
    Query PyPI for the latest version of pkg_name.

    Args:
        pkg_name (str): The name of the package on PyPI.

    Returns:
        Optional[Version]: The latest version as a packaging.version.Version object,
                           or None if not found or an error occurred.
    """
    try:
        url = f"{PYPI_BASE_URL}/{pkg_name}/json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        latest_str = data["info"]["version"]
        return parse_version(latest_str)
    except Exception as exc:
        logging.warning(f"Could not get PyPI version for {pkg_name}: {exc}")
        return None


def get_latest_conda_version(pkg_name: str, from_channel: str = DEFAULT_CONDA_CHANNEL) -> Optional[Version]:
    """
    Query Conda for the latest version of pkg_name using `conda search`.

    Args:
        pkg_name (str): The name of the package on Conda.
        from_channel (str): The Conda channel to search from.
                                      Defaults to DEFAULT_CONDA_CHANNEL.

    Returns:
        Optional[Version]: The latest version as a packaging.version.Version object,
                           or None if not found or an error occurred.
    """
    try:
        resp = requests.get(f"{DEFAULT_CONDA_CHANNEL}/linux-64/repodata.json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        versions = [parse_version(pkg["version"]) for pkg in data["packages"].values() if pkg["name"] == pkg_name]
        return max(versions) if versions else None
    except Exception as e:
        logging.warning(f"Failed to get Conda version for {pkg_name}: {e}")
        return None


def bump_upper_bound(latest: Version) -> str:
    """
    Compute the next major version's lower bound based on the latest version.

    Args:
        latest (Version): The latest version of the package.

    Returns:
        str: The next major version as a string (e.g., "5" for version "4.32.1").
    """
    major = latest.major
    return f"{major + 1}"


def parse_version_requirements(req: str) -> Optional[tuple[str, str]]:
    """
    Parse the version_requirements string using packaging.requirements.

    Expects format '>=a,<b' or similar Python version specifiers.

    Args:
        req (str): The version requirement string.

    Returns:
        Optional[Tuple[str, str]]: A tuple containing the lower and upper bounds,
                                  or None if the format doesn't match expectations.
    """
    try:
        # Create a dummy requirement to parse the version specifiers
        # packaging.requirements expects a package name, so we add a dummy one
        parsed = Requirement("dummy" + req)

        lower_bound = None
        upper_bound = None

        for spec in parsed.specifier:
            if spec.operator == ">=":
                lower_bound = str(spec.version)
            elif spec.operator == "<":
                upper_bound = str(spec.version)

        if lower_bound and upper_bound:
            return lower_bound, upper_bound

    except Exception as e:
        # Handle any parsing errors
        logging.warning(f"Failed to parse version requirements: {req} - {e}")
        return None

    return None


def update_version_requirement(req: str, latest_version: Version) -> Optional[str]:
    """
    Update the upper bound of the version requirement.

    If the requirement matches '>=a, <b', it updates to '>=a, <c',
    where 'c' is the next major version.

    Args:
        req (str): The original version requirement string.
        latest_version (Version): The latest version of the package.

    Returns:
        Optional[str]: The updated requirement string, or None if the format doesn't match.
    """
    parsed = parse_version_requirements(req)
    if parsed:
        lower_bound, _ = parsed
        new_upper_bound = f"{bump_upper_bound(latest_version)}"
        return f">={lower_bound},<{new_upper_bound}"
    return None


def update_requirements(
    package_entry: dict[str, str],
    latest_pypi: Optional[Version],
    latest_conda: Optional[Version],
    dry_run: bool,
) -> bool:
    """
    Update the version_requirements* fields in the package_entry dict in-place.

    If they exist and match the expected format, update the upper bound based on
    the latest version in PyPI/Conda.

    Args:
        package_entry (dict): The package entry from requirements.yml.
        latest_pypi (Optional[Version]): The latest PyPI version.
        latest_conda (Optional[Version]): The latest Conda version.
        dry_run (bool): If True, only print the updates without modifying the file.

    Returns:
        bool: True if an update is made, False otherwise.
    """
    updated = False

    if not latest_pypi and not latest_conda:
        logging.warning(f"No latest version found for PyPI or Conda for package '{package_entry.get('name')}'")
        return updated

    # Update PyPI version requirements
    if latest_pypi:
        if "version_requirements_pypi" in package_entry:
            original_req = package_entry["version_requirements_pypi"]
            updated_req = update_version_requirement(original_req, latest_pypi)
            if updated_req and updated_req != original_req:
                pkg_name = package_entry.get("name_pypi", package_entry.get("name", ""))
                if dry_run:
                    logging.info(
                        f"[DRY-RUN] PyPI package '{pkg_name}' latest PYPI version {latest_pypi}: "
                        f"'{original_req}' -> '{updated_req}'"
                    )
                else:
                    package_entry["version_requirements_pypi"] = updated_req
                    logging.info(f"Updated PyPI package '{pkg_name}': '{original_req}' -> '{updated_req}'")
                updated = True
        elif "version_requirements" in package_entry:
            original_req = package_entry["version_requirements"]
            updated_req = update_version_requirement(original_req, latest_pypi)
            if updated_req and updated_req != original_req:
                pkg_name = package_entry.get("name", package_entry.get("name_pypi", ""))
                if dry_run:
                    logging.info(
                        f"[DRY-RUN] PyPI/Conda package '{pkg_name}' latest PYPI version {latest_pypi}: "
                        f"'{original_req}' -> '{updated_req}'"
                    )
                else:
                    package_entry["version_requirements"] = updated_req
                    logging.info(f"Updated PyPI/Conda package '{pkg_name}': '{original_req}' -> '{updated_req}'")
                updated = True

    # Update Conda version requirements
    # Note: Conda version requirements will override PyPI version requirements on "version_requirements" field
    if latest_conda:
        if "version_requirements_conda" in package_entry:
            original_req = package_entry["version_requirements_conda"]
            updated_req = update_version_requirement(original_req, latest_conda)
            if updated_req and updated_req != original_req:
                pkg_name = package_entry.get("name_conda", package_entry.get("name", ""))
                if dry_run:
                    logging.info(
                        f"[DRY-RUN] Conda package '{pkg_name}' latest Conda version {latest_conda}: "
                        f"'{original_req}' -> '{updated_req}'"
                    )
                else:
                    package_entry["version_requirements_conda"] = updated_req
                    logging.info(f"Updated Conda package '{pkg_name}': '{original_req}' -> '{updated_req}'")
                updated = True
        elif "version_requirements" in package_entry:
            original_req = package_entry["version_requirements"]
            updated_req = update_version_requirement(original_req, latest_conda)
            if updated_req and updated_req != original_req:
                pkg_name = package_entry.get("name", package_entry.get("name_conda", ""))
                if dry_run:
                    logging.info(
                        f"[DRY-RUN] PyPI/Conda package '{pkg_name}' latest Conda version {latest_conda}: "
                        f"'{original_req}' -> '{updated_req}'"
                    )
                else:
                    package_entry["version_requirements"] = updated_req
                    logging.info(f"Updated PyPI/Conda package '{pkg_name}': '{original_req}' -> '{updated_req}'")
                updated = True

    return updated


def main() -> None:
    """
    Main function to update upper bounds of version_requirements in requirements.yml.

    Parses command-line arguments, processes each package entry, and updates
    the version requirements accordingly.
    """
    parser = argparse.ArgumentParser(description="Update upper bounds of version_requirements in requirements.yml")
    parser.add_argument("requirements_yml", help="Path to requirements.yml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check latest versions and print updates without modifying the file",
    )
    args = parser.parse_args()

    requirements_yml = args.requirements_yml

    if not os.path.exists(requirements_yml):
        logging.error(f"File not found: {requirements_yml}")
        sys.exit(1)

    with open(requirements_yml) as f:
        data = yaml.load(f)

    if not isinstance(data, list):
        logging.error("The top-level structure of requirements.yml is not a list.")
        sys.exit(2)

    updated_count = 0
    for pkg_entry in data:
        name = pkg_entry.get("name")
        name_pypi = pkg_entry.get("name_pypi")
        name_conda = pkg_entry.get("name_conda")

        # Determine package names for PyPI and Conda
        pypi_pkg_name = name_pypi if name_pypi else name
        conda_pkg_name = name_conda if name_conda else name

        # Fetch latest versions
        latest_pypi = get_latest_pypi_version(pypi_pkg_name) if pypi_pkg_name else None
        conda_channel = pkg_entry.get("from_channel", DEFAULT_CONDA_CHANNEL) if conda_pkg_name else None
        latest_conda = get_latest_conda_version(conda_pkg_name, from_channel=conda_channel) if conda_pkg_name else None

        # Update requirements
        updated = update_requirements(pkg_entry, latest_pypi, latest_conda, args.dry_run)
        if updated:
            updated_count += 1

    if not args.dry_run:
        with open(requirements_yml, "w") as f:
            yaml.dump(data, f)

    if args.dry_run:
        logging.info(f"Dry run complete! Found {updated_count} potential updates.")
    else:
        logging.info(f"Done! Updated the version requirements for {updated_count} packages.")


if __name__ == "__main__":
    main()

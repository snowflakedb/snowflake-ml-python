#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from typing import Match, Optional


def get_targets_with_cquery(target_pattern: str, bazel_path: str = "bazel") -> list[tuple[str, str, list[str]]]:
    """
    Get py_test targets with their feature areas and source files using bazel query.

    Args:
        target_pattern: bazel target pattern to search (e.g., "//tests/integ/...")
        bazel_path: path to bazel executable

    Returns:
        List of tuples with the following elements:
        - target_name (str): full bazel target name (e.g., "//tests/integ:my_test")
        - feature_area (str): feature area extracted from tags (e.g., "jobs", "model_registry")
        - source_files (list[str]): list of python source file paths for the target
    """
    targets_info = []

    try:
        # Query for py_test targets
        cmd = [bazel_path, "query", f'kind("py_test", {target_pattern})', "--output=label"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        for target in targets:
            feature_area = get_feature_area(target, bazel_path)
            if feature_area:
                source_files = get_source_files(target, bazel_path)
                if source_files:
                    targets_info.append((target, feature_area, source_files))

        return targets_info

    except subprocess.CalledProcessError:
        return []
    except Exception:
        return []


def get_feature_area(target: str, bazel_path: str) -> Optional[str]:
    """Extract feature area from target."""
    try:
        cmd = [bazel_path, "query", f"attr('tags', 'feature:.*', {target})", "--output=build"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Look for feature: tags in the build output
        for line in result.stdout.splitlines():
            if "feature:" in line:
                match = re.search(r'feature:([^"\']+)', line)
                if match:
                    return match.group(1)

        return None

    except subprocess.CalledProcessError:
        return None


def get_source_files(target: str, bazel_path: str = "bazel") -> list[str]:
    """Get source files for target."""
    try:
        cmd = [bazel_path, "query", f"attr('srcs', '.*', {target})", "--output=build"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        source_files = []
        target_dir = target.replace("//", "").split(":")[0]

        # Extract .py files from srcs attribute
        for line in result.stdout.splitlines():
            if "srcs = [" in line or '.py"' in line:
                py_files = re.findall(r'"([^"]*\.py)"', line)
                for py_file in py_files:
                    if not py_file.startswith("//"):
                        # Relative path, add target directory
                        source_files.append(f"{target_dir}/{py_file}")
                    else:
                        # Absolute path, convert to relative
                        source_files.append(py_file.replace("//", "").replace(":", "/"))

        return source_files

    except subprocess.CalledProcessError:
        return []


def process_test_file(source_file: str, feature_area: str) -> None:
    """
    Process a python test file to add pytest feature area marks.
    * adds "import pytest" if not already present
    * adds @pytest.mark.feature_area_<feature_area> decorators to test classes or functions

    Args:
        source_file: path to the python test file to process
        feature_area: feature area name (e.g., "jobs", "model_registry") used in pytest marks
    """
    if not os.path.exists(source_file):
        return

    try:
        with open(source_file, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Add pytest import at the top with other imports if not present
        if "import pytest" not in content:
            lines = content.split("\n")
            # Find the first import line to insert pytest import there
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("from ") or stripped.startswith("import "):
                    insert_idx = i
                    break
                elif stripped and not stripped.startswith("#"):
                    # If we hit non-comment, non-import content, insert at current position
                    insert_idx = i
                    break

            lines.insert(insert_idx, "import pytest")
            content = "\n".join(lines)

        # Add pytest marks using
        mark = f"@pytest.mark.feature_area_{feature_area}"

        # Skip if mark is already present
        if mark in content:
            return

        # Mark test classes - match classes that start with "Test" OR end with "Test"
        class_pattern = r"(\n+)(\s*)(class (?:Test\w*|\w*Test).*?:)$"
        class_marked = False

        def add_class_mark(match: Match[str]) -> str:
            nonlocal class_marked
            class_marked = True
            preceding_newlines = match.group(1)
            indent = match.group(2)
            definition = match.group(3)
            # Preserve original spacing - add mark right before class definition
            return f"{preceding_newlines}{indent}{mark}\n{indent}{definition}"

        new_content = re.sub(class_pattern, add_class_mark, content, flags=re.MULTILINE)
        if new_content != content:
            content = new_content

        # Only mark test functions if no test class was marked
        if not class_marked:
            func_pattern = r"^(\s*)(def test_\w*.*?:)$"

            def add_func_mark(match: Match[str]) -> str:
                indent = match.group(1)
                definition = match.group(2)
                return f"{indent}{mark}\n{indent}{definition}"

            new_content = re.sub(func_pattern, add_func_mark, content, flags=re.MULTILINE)
            if new_content != content:
                content = new_content

        # Only write if changes were made
        if content != original_content:
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(content)

    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add pytest marks to test files based on Bazel py_test target feature tags"
    )
    parser.add_argument("--targets", default="//...", help="Bazel target pattern to process (default: //...)")
    parser.add_argument("--bazel-path", default="bazel", help="Path to bazel executable")

    args = parser.parse_args()

    try:
        targets_info = get_targets_with_cquery(args.targets, args.bazel_path)

        if not targets_info:
            return 0

        for _target_name, feature_area, source_files in targets_info:
            for source_file in source_files:
                process_test_file(source_file, feature_area)

        return 0

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())

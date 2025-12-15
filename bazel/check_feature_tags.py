#!/usr/bin/env python3
"""
Utility to find py_test targets without feature area tags.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Valid feature areas (should match py_rules.bzl)
VALID_FEATURE_AREAS = [
    "model_registry",
    "feature_store",
    "jobs",
    "observability",
    "cortex",
    "core",
    "modeling",
    "model_serving",
    "data",
    "none",
]

FEATURE_TAG_FORMAT = 'tags = ["feature:<feature_area>"]'


def get_changed_build_files() -> list[str]:
    """Get list of changed BUILD.bazel files from git."""
    try:
        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"], capture_output=True, text=True, check=True
        )

        changed_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Filter for BUILD.bazel files
        build_files = [f for f in changed_files if f.endswith("BUILD.bazel")]

        return build_files

    except subprocess.CalledProcessError:
        # If git command fails, return empty list
        return []


def find_untagged_py_tests(root_dir: str = ".", exclude_experimental: bool = True) -> list[tuple[str, str, int, str]]:
    """
    Find py_test targets that don't have feature area tags.

    Args:
        root_dir: Root directory to search from
        exclude_experimental: Whether to exclude experimental directories

    Returns:
        List of tuples: (file_path, target_name, line_number)
    """
    untagged_targets = []

    for root, _dirs, files in os.walk(root_dir):
        # Skip build directories and other artifacts
        if any(skip_dir in root for skip_dir in ["build/", "venv/", ".git/", "__pycache__/"]):
            continue

        # Skip experimental directories if requested
        if exclude_experimental and "experimental" in root:
            continue

        if "BUILD.bazel" in files:
            build_file = Path(root) / "BUILD.bazel"
            untagged_in_file = find_untagged_in_file(build_file)
            untagged_targets.extend(untagged_in_file)

    return untagged_targets


def find_untagged_in_file(file_path: Path) -> list[tuple[str, str, int, str]]:
    """
    Find py_test targets with missing or invalid feature area tags in a specific BUILD.bazel file.

    Args:
        file_path: Path to BUILD.bazel file

    Returns:
        List of tuples: (file_path, target_name, line_number, issue_type)
        where issue_type is either "missing" or "invalid:<actual_area>"
    """
    try:
        with open(file_path) as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return []

    untagged_targets = []
    lines = content.split("\n")

    # Find py_test blocks
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip commented lines
        if line.startswith("#"):
            i += 1
            continue

        # Look for py_test(
        py_test_match = re.match(r"py_test\(\s*$", line)
        if py_test_match:
            target_name = None
            feature_area = None
            start_line = i + 1  # 1-indexed

            # Parse the py_test block
            j = i + 1
            paren_count = 1
            while j < len(lines) and paren_count > 0:
                block_line = lines[j]

                # Count parentheses to find block end
                paren_count += block_line.count("(") - block_line.count(")")

                # Extract target name
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block_line)
                if name_match:
                    target_name = name_match.group(1)

                # Extract feature area from feature tag
                feature_match = re.search(r'"feature:([^"]+)"', block_line)
                if feature_match:
                    feature_area = feature_match.group(1)

                j += 1

            # Check for issues with feature tags
            if target_name:
                if feature_area is None:
                    # No feature tag found
                    untagged_targets.append((str(file_path), target_name, start_line, "missing"))
                elif feature_area not in VALID_FEATURE_AREAS:
                    # Invalid feature area
                    untagged_targets.append((str(file_path), target_name, start_line, f"invalid:{feature_area}"))

        i += 1

    return untagged_targets


def check_build_files_for_py_test_tags(build_files: list[str]) -> list[tuple[str, str, int, str]]:
    """Check specific BUILD.bazel files for py_test targets with missing or invalid feature area tags."""
    untagged_targets = []

    for build_file in build_files:
        file_path = Path(build_file)
        if file_path.exists():
            untagged_in_file = find_untagged_in_file(file_path)
            untagged_targets.extend(untagged_in_file)

    return untagged_targets


def print_utility_output(untagged_targets: list[tuple[str, str, int, str]]) -> int:
    """Print output in utility mode."""
    if not untagged_targets:
        print("All py_test targets have feature area tags!")
        return 0

    print(f"\nFound {len(untagged_targets)} py_test targets without feature area tags:")
    print("-" * 80)

    # Group by file for better readability
    by_file: dict[str, list[tuple[str, int, str]]] = {}
    for file_path, target_name, line_num, issue_type in untagged_targets:
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append((target_name, line_num, issue_type))

    for file_path, targets in sorted(by_file.items()):
        print(f"\n📁 {file_path}")
        for target_name, line_num, issue_type in targets:
            if issue_type == "missing":
                print(f"   • {target_name} (line {line_num}) - missing feature tag")
            else:
                invalid_area = issue_type.split(":", 1)[1]
                print(f"   • {target_name} (line {line_num}) - invalid feature area: {invalid_area}")

    print("\nTo fix these issues:")
    print("1. Add a feature area tag to each py_test target")
    print(f"2. Choose the appropriate feature area from: {', '.join(VALID_FEATURE_AREAS)}")
    print('3. Use format: tags = ["feature:<area>"]')

    return 1


def print_precommit_output(untagged_targets: list[tuple[str, str, int, str]], mode: str = "precommit") -> int:
    """Print output in pre-commit mode with detailed help."""
    if not untagged_targets:
        print("All py_test targets have feature area tags!")
        return 0

    # Report untagged targets
    print(f"\nFound {len(untagged_targets)} py_test targets without feature area tags:")
    print("-" * 80)

    # Group by file
    by_file: dict[str, list[tuple[str, int, str]]] = {}
    for file_path, target_name, line_num, issue_type in untagged_targets:
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append((target_name, line_num, issue_type))

    for file_path, targets in sorted(by_file.items()):
        print(f"\n📁 {file_path}")
        for target_name, line_num, issue_type in targets:
            if issue_type == "missing":
                print(f"   • {target_name} (line {line_num}) - missing feature tag")
            else:  # invalid:area_name
                invalid_area = issue_type.split(":", 1)[1]
                print(f"   • {target_name} (line {line_num}) - invalid feature area: {invalid_area}")

    print("\n" + "=" * 60)
    print("FEATURE AREA TAGGING HELP")
    print("=" * 60)
    print("\nAll py_test targets must include a feature area tag.")
    print('Format: tags = ["feature:<feature_area>"]')
    print(f"\nValid feature areas: {', '.join(VALID_FEATURE_AREAS)}")

    return 1


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check for py_test targets without feature area tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check entire repository (excluding experimental)
  bazel run //bazel:check_feature_tags

  # Include experimental directories
  bazel run //bazel:check_feature_tags -- --include-experimental

  # Pre-commit mode (check only changed files)
  bazel run //bazel:check_feature_tags -- --precommit

  # Check specific files
  bazel run //bazel:check_feature_tags -- --files path/to/BUILD.bazel another/BUILD.bazel
        """,
    )

    parser.add_argument("--include-experimental", action="store_true", help="Include experimental directories in scan")

    parser.add_argument("--precommit", action="store_true", help="Run in pre-commit mode (check only git staged files)")

    parser.add_argument("--files", nargs="*", help="Check specific BUILD.bazel files")

    parser.add_argument("--verbose", action="store_true", help="Show detailed output with feature area descriptions")

    # Positional arguments for pre-commit compatibility
    parser.add_argument("positional_files", nargs="*", help="BUILD.bazel files to check (for pre-commit compatibility)")

    args = parser.parse_args()

    # Change to repo root if running in precommit mode
    if args.precommit:
        try:
            repo_root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
            ).stdout.strip()
            os.chdir(repo_root)
            # Add repo root to Python path
            sys.path.insert(0, repo_root)
        except subprocess.CalledProcessError:
            print("❌ Error: Not in a git repository")
            return 1

    # Combine --files and positional files arguments
    files_to_check = []
    if args.files:
        files_to_check.extend(args.files)
    if args.positional_files:  # positional files argument
        files_to_check.extend(args.positional_files)

    # Remove duplicates while preserving order
    files_to_check = list(dict.fromkeys(files_to_check))

    if files_to_check:
        print(f"   Checking {len(files_to_check)} specified files...")
        untagged_targets = check_build_files_for_py_test_tags(files_to_check)
    elif args.precommit:
        changed_build_files = get_changed_build_files()

        if not changed_build_files:
            untagged_targets = []
        else:
            print(f"   Checking {len(changed_build_files)} changed BUILD.bazel files...")
            untagged_targets = check_build_files_for_py_test_tags(changed_build_files)
    else:
        # Check entire repository
        exclude_experimental = not args.include_experimental
        if exclude_experimental:
            print("   (excluding experimental directories)")
        untagged_targets = find_untagged_py_tests(exclude_experimental=exclude_experimental)

    # Print results
    if args.precommit or args.verbose:
        return print_precommit_output(untagged_targets, mode="precommit" if args.precommit else "verbose")
    else:
        return print_utility_output(untagged_targets)


if __name__ == "__main__":
    sys.exit(main())

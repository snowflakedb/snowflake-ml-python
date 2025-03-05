import argparse
import sys
from importlib.machinery import SourceFileLoader

optional_dependency_groups_bzl = SourceFileLoader(
    "optional_dependency_groups_bzl", "bazel/platforms/optional_dependency_groups.bzl"
).load_module()


def update_bazelrc(bazelrc_path: str) -> None:
    with open(bazelrc_path) as f:
        lines = f.readlines()

    for group in optional_dependency_groups_bzl.OPTIONAL_DEPENDENCY_GROUPS.keys():
        lines.append("\n")
        lines.append(
            f"build:_{group} "
            f"--platforms //bazel/platforms:{group}_conda_env "
            f"--host_platform //bazel/platforms:{group}_conda_env "
            f"--repo_env=BAZEL_CONDA_ENV_NAME={group}\n"
        )
        lines.append(f"cquery:{group} --config=_{group}\n")
        lines.append(f"build:{group} --config=_{group}\n")
        lines.append(f"test:{group} --config=_{group}\n")
        lines.append(f"run:{group} --config=_{group}\n")

    sys.stdout.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bazelrc_path", help="Path to the .bazelrc file to update")
    args = parser.parse_args()
    update_bazelrc(args.bazelrc_path)

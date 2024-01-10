import sys

try:
    from mypy.main import main
except ImportError as e:
    raise ImportError(
        f"Unable to import mypy. Make sure mypy is added to the bazel conda environment. Actual error: {e}"
    )

if __name__ == "__main__":
    main(stdout=sys.stdout, stderr=sys.stderr)

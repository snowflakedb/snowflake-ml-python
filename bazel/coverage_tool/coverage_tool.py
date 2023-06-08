"""This is a wrapper to the coverage tool.
It injects a --ignore-errors argument when called to generate a coverage report, to avoid bazel fails when running
coverage tool to collect coverage report on a source code file that does not exist, for example, zip-imported source.
"""
import re
import sys

try:
    from coverage.cmdline import main
except ImportError as e:
    raise ImportError(
        f"Unable to import coverage. Make sure coverage is added to the bazel conda environment. Actual error: {e}"
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Too few arguments.")
    # This line is from the original coverage entrypoint.
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])

    action, options = sys.argv[1], sys.argv[2:]
    if action in ["report", "html", "xml", "json", "lcov", "annotate"]:
        options.insert(0, "--ignore-errors")
    args = [action] + options
    sys.exit(main(args))

import argparse
import json
import os
import subprocess
import sys
import tempfile

MYPY_ENTRYPOINT_CODE = """
import sys

try:
    from mypy.main import main
except ImportError as e:
    raise ImportError(
        f"Unable to import mypy. Make sure mypy is added to the bazel conda environment. Actual error: {{e}}"
    )

if __name__ == "__main__":
    main(stdout=sys.stdout, stderr=sys.stderr)

"""

MYPY_CACHE_DIR = ".mypy_cache"


def mypy_checker() -> None:
    # To parse the arguments that bazel provides.
    parser = argparse.ArgumentParser(
        # Without this, the second path documented in main below fails.
        fromfile_prefix_chars="@"
    )
    parser.add_argument("--out")
    parser.add_argument("--persistent_worker", action="store_true")

    args = parser.parse_args()

    os.makedirs(MYPY_CACHE_DIR, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".py") as mypy_entrypoint:
        mypy_entrypoint.write(MYPY_ENTRYPOINT_CODE.encode())
        mypy_entrypoint.flush()
        first_run = True
        while args.persistent_worker or first_run:
            data = sys.stdin.readline()
            req = json.loads(data)
            mypy_args = req["arguments"]
            mypy_args = ["--cache-dir", MYPY_CACHE_DIR] + mypy_args
            process = subprocess.Popen(
                # We use this to make sure we are invoking mypy that is installed in the same environment of the current
                # Python.
                [sys.executable, mypy_entrypoint.name] + mypy_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process.wait()
            text, err = process.communicate()
            message = text.decode() + err.decode()
            with open(args.out, "w") as output:
                output.write(message)
            sys.stderr.flush()
            sys.stdout.write(
                json.dumps(
                    {
                        "exitCode": process.returncode,
                        "output": message,
                        "requestId": req.get("requestId", 0),
                    }
                )
            )
            sys.stdout.flush()
            first_run = False


if __name__ == "__main__":
    mypy_checker()

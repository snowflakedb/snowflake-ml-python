import os
import sys

from build.__main__ import main

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    pyproject_toml_path = sys.argv[1]
    src_path = sys.argv[2]
    with open(pyproject_toml_path, encoding="utf-8") as rf:
        with open(os.path.join(src_path, "pyproject.toml"), "w", encoding="utf-8") as wf:
            wf.write(rf.read())
    main(sys.argv[2:])

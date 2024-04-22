import os
import subprocess
import sys
from subprocess import CalledProcessError

"""Wraps `sphinx-build`, all arguments are passed on to `sphinx-build`.
"""

if __name__ == "__main__":
    # Seemingly unused imports; but is to overcome error with namespaced Python packages (in sphinxcontrib.*) that would
    # otherwise not be importable.
    import sphinxcontrib  # noqa: F401

    # 'Sniff' the sources input directory as a way to know what it would be a configuration-time in conf.py.
    # Needed for pointing to custom CSS/JS.
    os.environ["BAZEL_SPHINX_INPUT_DIR"] = os.path.abspath(sys.argv[-2])

    for k, paths in os.environ.items():
        path_list = paths.split(":")
        os.environ[k] = ":".join([os.path.abspath(path) if os.path.exists(path) else path for path in path_list])

    # Monkey patch the bugged imgpath method
    from sphinx.builders import Builder
    from sphinx.cmd.build import main
    from sphinx.ext import imgmath

    def compile_math(latex: str, builder: Builder) -> str:
        """Compile LaTeX macros for math to DVI."""
        tempdir = imgmath.ensure_tempdir(builder)
        filename = os.path.join(tempdir, "math.tex")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)

        imgmath_latex_name = os.path.basename(builder.config.imgmath_latex)

        command = [builder.config.imgmath_latex]
        if imgmath_latex_name != "tectonic":
            command.append("--interaction=nonstopmode")
        # add custom args from the config file
        command.extend(builder.config.imgmath_latex_args)
        command.append(f"--output-directory={tempdir}")
        command.append("--output-format=dvi")
        command.append(filename)

        try:
            subprocess.run(command, capture_output=True, cwd=tempdir, check=True, encoding="ascii")
            if imgmath_latex_name in {"xelatex", "tectonic"}:
                return os.path.join(tempdir, "math.xdv")
            else:
                return os.path.join(tempdir, "math.dvi")
        except OSError as exc:
            imgmath.logger.warning(
                ("LaTeX command %r cannot be run (needed for math " "display), check the imgmath_latex setting"),
                builder.config.imgmath_latex,
            )
            raise imgmath.InvokeError from exc
        except CalledProcessError as exc:
            msg = "latex exited with error"
            raise imgmath.MathExtError(msg, exc.stderr, exc.stdout) from exc

    imgmath.compile_math = compile_math

    # Monkey patch ends

    sys.exit(main(sys.argv[1:]))

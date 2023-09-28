"Define Sphinx build rules"

load("@//bazel:py_rules.bzl", "py_binary")

# Borrowed from Rules Go, licensed under Apache 2.
# https://github.com/bazelbuild/rules_go/blob/67f44035d84a352cffb9465159e199066ecb814c/proto/compiler.bzl#L72
def _rel_path_from_workspace_root(file):
    path = file.path
    root = file.root.path
    ws = file.owner.workspace_root
    if path.startswith(root):
        path = path[len(root):]
    if path.startswith("/"):
        path = path[1:]
    if path.startswith(ws):
        path = path[len(ws):]
    if path.startswith("/"):
        path = path[1:]
    return path

def _sphinx_docs_impl(ctx):
    sphinx_input = []

    for include in ctx.files.includes:
        copied_file = ctx.actions.declare_file("sphinx_input/" + _rel_path_from_workspace_root(include))
        ctx.actions.run(
            outputs = [copied_file],
            inputs = [include],
            executable = "cp",
            arguments = [include.path, copied_file.path],
            progress_message = "Copying include %s to %s in sphinx input directory" % (include.path, copied_file.path),
            mnemonic = "CopyInclude",
        )
        sphinx_input.append(copied_file)

    for src in ctx.files.srcs:
        if src.is_source:
            # rst_path becomes the rel path from the build file
            rst_path = src.path.split(src.owner.package)[-1].strip("/")

            # This is declared as file, but actually a directory will be created at that path with the recursive copy
            # action. If we would declare a directory, it would be created already by Bazel.
            copied_file = ctx.actions.declare_file("sphinx_input/" + rst_path)
            ctx.actions.run(
                outputs = [copied_file],
                inputs = [src],
                executable = "cp",
                arguments = [src.path, copied_file.path],
                progress_message = "Copying src %s to %s in sphinx input directory" % (src.path, copied_file.path),
                mnemonic = "CopySrc",
            )
            sphinx_input.append(copied_file)

        else:
            rst_path = _rel_path_from_workspace_root(src)
            if rst_path.endswith(src.owner.name):
                rst_path = rst_path[:-len(src.owner.name)]

            # This is declared as file, but actually a directory will be created at that path with the recursive copy
            # action. If we would declare a directory, it would be created already by Bazel.
            copied_file = ctx.actions.declare_file("sphinx_input/api/" + rst_path)
            src_path_rst = "/".join([src.path, rst_path])
            ctx.actions.run(
                outputs = [copied_file],
                inputs = [src],
                executable = "cp",
                arguments = ["-r", src_path_rst, copied_file.path],
                progress_message = "Copying generated %s to %s in sphinx input directory" % (src.path, copied_file.path),
                mnemonic = "CopyGenSrc",
            )
            sphinx_input.append(copied_file)

    sphinx_src_dir_path = sphinx_input[0].path.split("sphinx_input/")[0] + "sphinx_input/"

    sphinx_output_dir = ctx.actions.declare_directory("html")
    ctx.actions.run(
        inputs = sphinx_input + [ctx.file.conf],
        outputs = [sphinx_output_dir],
        executable = ctx.executable.sphinx_main,
        arguments = [
            "-c",  # Alternative configuration directory, to not have a conf.py file along with content sources.
            ctx.file.conf.path,
            "-q",  # Only warnings and errors are written to standard error.
            "-E",  # Rebuild completely always, don't bother saving states for incremental builds.
            "-n",  # Fail (abort) on missing references.
            # "-W",  # Turn warning into errors ...
            "--keep-going",  # ..., but don't stop at the first error to list all errors.
            "-b",  # builder mode ('html')
            "html",
            # "-vv",  # Uncomment this to enable debug information output
            sphinx_src_dir_path,
            sphinx_output_dir.path,
        ],
        progress_message = "Building documentation",
        mnemonic = "SphinxDoc",
    )

    return [DefaultInfo(files = depset([sphinx_output_dir]))]

_sphinx_docs = rule(
    attrs = {
        "conf": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "includes": attr.label_list(
            allow_files = True,
        ),
        "sphinx_main": attr.label(
            executable = True,
            cfg = "exec",
        ),
        "srcs": attr.label_list(
            mandatory = True,
        ),
    },
    implementation = _sphinx_docs_impl,
)

def sphinx_docs(name, deps, extensions = [], **kwargs):
    sphinx_main_name = name + "_sphinx_main"
    py_binary(
        name = sphinx_main_name,
        srcs = ["@rules_sphinx//:sphinx_main.py"],
        visibility = ["//visibility:public"],
        main = "sphinx_main.py",
        deps = deps + extensions,
    )
    _sphinx_docs(name = name, sphinx_main = sphinx_main_name, **kwargs)

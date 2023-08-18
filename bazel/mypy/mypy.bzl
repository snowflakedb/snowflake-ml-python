"Public API"

load("//bazel/mypy:rules.bzl", "MyPyStubsInfo")

MyPyAspectInfo = provider(
    """This is an aspect attaching to the original Python build graph to type-checking Python source files.
    For every target it collects all transitive dependencies as well as direct sources and use symbol link to create
    a folder ends with .mypy_runfiles. Mypy will be invoked to check the direct sources.

    This aspect uses persistent worker to make full use of mypy's cache which is defined in main.py in the same
    directory. The mypy cache will be put into bazel's execroot/SnowML/,mypy_cache .""",
    fields = {
        "exe": "Used to pass the rule implementation built exe back to calling aspect.",
        "args": "Used to pass the arguments sent to mypy executable.",
        "runfiles": "Used to pass the inputs file for mypy executable.",
        "out": "Used to pass the dummy output file back to calling aspect.",
    },
)

# Switch to True only during debugging and development.
# All releases should have this as False.
DEBUG = False

VALID_EXTENSIONS = ["py", "pyi"]

DEFAULT_ATTRS = {
    "_mypy_cli": attr.label(
        default = Label("//bazel/mypy:mypy"),
        executable = True,
        cfg = "exec",
    ),
    "_mypy_config": attr.label(
        default = Label("//:mypy.ini"),
        allow_single_file = True,
    ),
}

def _is_external_dep(dep):
    return dep.label.workspace_root.startswith("external/")

def _is_external_src(src_file):
    return src_file.path.startswith("external/")

def _extract_srcs(srcs):
    direct_src_files = []
    for src in srcs:
        for f in src.files.to_list():
            if f.extension in VALID_EXTENSIONS:
                direct_src_files.append(f)
    return direct_src_files

def _extract_transitive_deps(deps):
    transitive_deps = []
    for dep in deps:
        if MyPyStubsInfo not in dep and PyInfo in dep and not _is_external_dep(dep):
            transitive_deps.append(dep[PyInfo].transitive_sources)
    return transitive_deps

def _extract_stub_deps(deps):
    # Need to add the .py files AND the .pyi files that are
    # deps of the rule
    stub_files = []
    for dep in deps:
        if MyPyStubsInfo in dep:
            for stub_srcs_target in dep[MyPyStubsInfo].srcs:
                for src_f in stub_srcs_target.files.to_list():
                    if src_f.extension == "pyi":
                        stub_files.append(src_f)
    return stub_files

def _mypy_rule_impl(ctx):
    base_rule = ctx.rule

    mypy_config_file = ctx.file._mypy_config

    direct_src_files = []
    transitive_srcs_depsets = []
    stub_files = []

    if hasattr(base_rule.attr, "srcs"):
        direct_src_files = _extract_srcs(base_rule.attr.srcs)

    if hasattr(base_rule.attr, "deps"):
        transitive_srcs_depsets = _extract_transitive_deps(base_rule.attr.deps)
        stub_files = _extract_stub_deps(base_rule.attr.deps)

    final_srcs_depset = depset(transitive = transitive_srcs_depsets +
                                            [depset(direct = direct_src_files)])
    src_files = [f for f in final_srcs_depset.to_list() if not _is_external_src(f)]
    if not src_files:
        return None

    out = ctx.actions.declare_file("%s_dummy_out" % ctx.rule.attr.name)
    runfiles_name = "%s.mypy_runfiles" % ctx.rule.attr.name

    # Compose a list of the files needed for use. Note that aspect rules can use
    # the project version of mypy however, other rules should fall back on their
    # relative runfiles.

    src_run_files = []
    direct_src_run_files = []
    stub_run_files = []

    for f in src_files + stub_files:
        run_file_path = runfiles_name + "/" + f.short_path
        run_file = ctx.actions.declare_file(run_file_path)
        ctx.actions.symlink(
            output = run_file,
            target_file = f,
        )
        if f in src_files:
            src_run_files.append(run_file)
        if f in direct_src_files:
            direct_src_run_files.append(run_file)
        if f in stub_files:
            stub_run_files.append(run_file)

    src_root_path = src_run_files[0].path
    src_root_path = src_root_path[0:(src_root_path.find(runfiles_name) + len(runfiles_name))]

    # arguments sent to mypy
    args = [
        "--enable-incomplete-features",
    ] + ["--package-root", src_root_path, "--config-file", mypy_config_file.path] + [f.path for f in direct_src_run_files]

    worker_arg_file = ctx.actions.declare_file(ctx.rule.attr.name + ".worker_args")
    ctx.actions.write(
        output = worker_arg_file,
        content = "\n".join(args),
    )

    return MyPyAspectInfo(
        exe = ctx.executable._mypy_cli,
        args = worker_arg_file,
        runfiles = src_run_files + stub_run_files + [mypy_config_file, worker_arg_file],
        out = out,
    )

def _mypy_aspect_impl(_, ctx):
    if (ctx.rule.kind not in ["py_binary", "py_library", "py_test", "mypy_test"] or
        ctx.label.workspace_root.startswith("external")):
        return []

    aspect_info = _mypy_rule_impl(
        ctx,
    )
    if not aspect_info:
        return []

    ctx.actions.run(
        outputs = [aspect_info.out],
        inputs = aspect_info.runfiles,
        tools = [aspect_info.exe],
        executable = aspect_info.exe,
        mnemonic = "MyPy",
        progress_message = "Type-checking %s" % ctx.label,
        execution_requirements = {
            "supports-workers": "1",
            "requires-worker-protocol": "json",
        },
        # out is required for worker to write the output.
        arguments = ["--out", aspect_info.out.path, "@" + aspect_info.args.path],
        use_default_shell_env = True,
    )
    return [
        OutputGroupInfo(
            mypy = depset([aspect_info.out]),
        ),
    ]

mypy_aspect = aspect(
    implementation = _mypy_aspect_impl,
    attr_aspects = ["deps"],
    attrs = DEFAULT_ATTRS,
)

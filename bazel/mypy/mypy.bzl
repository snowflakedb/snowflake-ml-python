"Public API"

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_skylib//lib:sets.bzl", "sets")
load("//bazel/mypy:rules.bzl", "MyPyStubsInfo")

MyPyAspectInfo = provider(
    "TODO: documentation",
    fields = {
        "exe": "Used to pass the rule implementation built exe back to calling aspect.",
        "out": "Used to pass the dummy output file back to calling aspect.",
    },
)

# Switch to True only during debugging and development.
# All releases should have this as False.
DEBUG = False

VALID_EXTENSIONS = ["py", "pyi"]

DEFAULT_ATTRS = {
    "_template": attr.label(
        default = Label("//bazel/mypy:mypy.sh.tpl"),
        allow_single_file = True,
    ),
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

def _sources_to_cache_map_triples(srcs):
    triples_as_flat_list = []
    for f in srcs:
        f_path = f.path
        triples_as_flat_list.extend([
            shell.quote(f_path),
            shell.quote("{}.meta.json".format(f_path)),
            shell.quote("{}.data.json".format(f_path)),
        ])
    return triples_as_flat_list

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

def _extract_imports(imports, label):
    # NOTE: Bazel's implementation of this for py_binary, py_test is at
    # src/main/java/com/google/devtools/build/lib/bazel/rules/python/BazelPythonSemantics.java
    mypypath_parts = []
    for import_ in imports:
        if import_.startswith("/"):
            # buildifier: disable=print
            print("ignoring invalid absolute path '{}'".format(import_))
        elif import_ in ["", "."]:
            mypypath_parts.append(label.package)
        else:
            mypypath_parts.append("{}/{}".format(label.package, import_))
    return mypypath_parts

def _mypy_rule_impl(ctx):
    base_rule = ctx.rule

    mypy_config_file = ctx.file._mypy_config

    mypypath_parts = []
    direct_src_files = []
    transitive_srcs_depsets = []
    stub_files = []

    if hasattr(base_rule.attr, "srcs"):
        direct_src_files = _extract_srcs(base_rule.attr.srcs)

    if hasattr(base_rule.attr, "deps"):
        transitive_srcs_depsets = _extract_transitive_deps(base_rule.attr.deps)
        stub_files = _extract_stub_deps(base_rule.attr.deps)

    if hasattr(base_rule.attr, "imports"):
        mypypath_parts = _extract_imports(base_rule.attr.imports, ctx.label)

    final_srcs_depset = depset(transitive = transitive_srcs_depsets +
                                            [depset(direct = direct_src_files)])
    src_files = [f for f in final_srcs_depset.to_list() if not _is_external_src(f)]
    if not src_files:
        return None

    mypypath_parts += [src_f.dirname for src_f in stub_files]
    mypypath = ":".join(mypypath_parts)

    out = ctx.actions.declare_file("%s_dummy_out" % ctx.rule.attr.name)
    exe = ctx.actions.declare_file(
        "%s_mypy_exe" % ctx.rule.attr.name,
    )

    # Compose a list of the files needed for use. Note that aspect rules can use
    # the project version of mypy however, other rules should fall back on their
    # relative runfiles.
    runfiles = ctx.runfiles(files = src_files + stub_files + [mypy_config_file])

    src_root_paths = sets.to_list(
        sets.make([f.root.path for f in src_files]),
    )

    ctx.actions.expand_template(
        template = ctx.file._template,
        output = exe,
        substitutions = {
            "{MYPY_EXE}": ctx.executable._mypy_cli.path,
            "{MYPY_ROOT}": ctx.executable._mypy_cli.root.path,
            "{CACHE_MAP_TRIPLES}": " ".join(_sources_to_cache_map_triples(src_files)),
            "{PACKAGE_ROOTS}": " ".join([
                "--package-root " + shell.quote(path or ".")
                for path in src_root_paths
            ]),
            "{SRCS}": " ".join([
                shell.quote(f.path)
                for f in src_files
            ]),
            "{VERBOSE_OPT}": "--verbose" if DEBUG else "",
            "{VERBOSE_BASH}": "set -x" if DEBUG else "",
            "{OUTPUT}": out.path if out else "",
            "{MYPYPATH_PATH}": mypypath if mypypath else "",
            "{MYPY_INI_PATH}": mypy_config_file.path,
        },
        is_executable = True,
    )

    return [
        DefaultInfo(executable = exe, runfiles = runfiles),
        MyPyAspectInfo(exe = exe, out = out),
    ]

def _mypy_aspect_impl(_, ctx):
    if (ctx.rule.kind not in ["py_binary", "py_library", "py_test", "mypy_test"] or
        ctx.label.workspace_root.startswith("external")):
        return []

    providers = _mypy_rule_impl(
        ctx
    )
    if not providers:
        return []

    info = providers[0]
    aspect_info = providers[1]

    ctx.actions.run(
        outputs = [aspect_info.out],
        inputs = info.default_runfiles.files,
        tools = [ctx.executable._mypy_cli],
        executable = aspect_info.exe,
        mnemonic = "MyPy",
        progress_message = "Type-checking %s" % ctx.label,
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

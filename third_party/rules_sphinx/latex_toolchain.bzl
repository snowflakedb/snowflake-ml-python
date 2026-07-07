"""snowml-local LaTeX toolchain registration for linux/aarch64 (ARM Linux).

The `bazel_latex` module (`prodrive_technologies_bazel_latex`) fetches aarch64-linux
TeX Live binaries but registers a LaTeX toolchain only for `aarch64-darwin`,
`x86_64-darwin`, and `x86_64-linux` — never `aarch64-linux`. Upstream
ProdriveTechnologies master has the same gap. As a result `bazel build //docs:docs`
fails at analysis on ARM Linux with:

    No matching toolchains found for @@..._bazel_latex~//:latex_toolchain_type
    (target //docs:docs_latex_engine)

This module defines an `aarch64-linux` toolchain so the docs build resolves natively
on ARM Linux. It mirrors the module's own (private) `_latex_toolchain_info` rule from
`@bazel_latex//:toolchain.bzl`, reusing the module's exported `LatexInfo` provider so
the `latex_engine` rule consumes it unchanged.

`biber` is absent from the aarch64-linux TeX Live build (the same reason the module
disables its `amd64-freebsd` toolchain) but is unused by the sphinx LaTeX engine
(`latex_engine.bzl` `_latex_impl` drives `luahbtex` + `luatex`/`dvisvgm`/`gsftopk`/
`kpsewhich`/`mktexlsr`/`kpsestat`/`kpseaccess`), so `biber` is stood in with `bibtex`.

TODO: remove this once `aarch64-linux` support lands in the `bazel_latex` module
(register the toolchain + make `biber` optional / add it to the aarch64-linux tarball).
"""

load("@bazel_latex//:toolchain.bzl", "LatexInfo")

def _latex_toolchain_info_impl(ctx):
    return [
        platform_common.ToolchainInfo(
            latexinfo = LatexInfo(
                biber = ctx.attr.biber,
                bibtex = ctx.attr.bibtex,
                dvisvgm = ctx.attr.dvisvgm,
                gsftopk = ctx.attr.gsftopk,
                kpsewhich = ctx.attr.kpsewhich,
                kpsestat = ctx.attr.kpsestat,
                luahbtex = ctx.attr.luahbtex,
                luatex = ctx.attr.luatex,
                mktexlsr = ctx.attr.mktexlsr,
                kpseaccess = ctx.attr.kpseaccess,
            ),
        ),
    ]

def _exe_attr():
    return attr.label(allow_single_file = True, cfg = "exec", executable = True)

latex_toolchain_info = rule(
    doc = "Reimplements bazel_latex's private _latex_toolchain_info using the exported LatexInfo provider.",
    attrs = {
        "biber": _exe_attr(),
        "bibtex": _exe_attr(),
        "dvisvgm": _exe_attr(),
        "gsftopk": _exe_attr(),
        "kpseaccess": _exe_attr(),
        "kpsestat": _exe_attr(),
        "kpsewhich": _exe_attr(),
        "luahbtex": _exe_attr(),
        "luatex": _exe_attr(),
        "mktexlsr": _exe_attr(),
    },
    implementation = _latex_toolchain_info_impl,
)

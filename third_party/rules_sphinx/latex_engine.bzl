"""
Rules to create latex engine

Credit: https://github.com/ProdriveTechnologies/bazel-latex/blob/master/latex.bzl
"""

LatexEngineInfo = provider(
    "Information about the result of a LaTeX compilation.",
    fields = {
        "deps": "depset of files the document depends on",
        "env": "The environment variables that will be used to run the latex engine.",
        "fmt": "The fmt file used by latex engine.",
        "fontmaps": "List of fontmaps that will be used by dvisvgm/dvipng toolchain.",
        "toolchain": "the toolchain being used.",
    },
)

def get_env(ctx, toolchain, files):
    """
    Set up environment variables common for all commands.

    Latex and a set of scripts and binaries in the tool suite
    makes use of a library, kpathsea.
    In general latex distributions are encouraged to follow the 'TDS'
    structure. And tools and script might make assumptions that the
    layout of directories respects that structure.

    But from our perspective, trying to shoehorn the partitioned
    repositories according
    (E.g. <sandbox>/bazel-latex/external/<ext_texlive_repo>/<files>)
    to the TDS creates an unnecessary complexity.
    Also kpathsea tries to be efficient about looking up files.
    So to derive if a folder is of interest, kpathsea checks if the number
    of files or folders in the current folder being inspected is greater
    than 2.
    Unfortunately symlinks are (currently) not counted. And Bazel makes heavy
    use of symlinks.

    However, kpathsea makes heavy use of environment variables (and
    ls-R database, IIRC).
    So we can work around this limitation. Also, by only adding the
    search paths of the files mapped to the environment variable we can reduce
    the search space, and reduce build times.
    https://tug.org/texinfohtml/kpathsea.html#Supported-file-formats,
    lists all environment variables one can set.

    Args:
      ctx: For accessing the inputs parameters.
      toolchain: The latex toolchain.
      files: all files that might be needed as part of the build.

    Returns:
      A list of commands to provide to the ctx.actions.run invocation
    """

    def list_unique_folders_from_file_ext(files, exts):
        directories = []
        for inp in files:
            dirname = inp.dirname
            valid = False
            if not exts:
                valid = True
            else:
                for ext in exts:
                    if inp.path.endswith(ext):
                        valid = True
            if valid and dirname not in directories:
                directories.append(dirname)
        return directories

    def setup_env_for(
            type_env_dict,
            env_var,
            files,
            extensions = [],
            post_additions = ""):
        search_folders = list_unique_folders_from_file_ext(
            files,
            extensions,
        )
        type_env_dict[env_var] = ".:{}{}".format(
            ":".join(search_folders),
            post_additions,
        )

    type_env = {}
    setup_env_for(type_env, "AFMFONTS", files, [".afm"])
    setup_env_for(type_env, "BIBINPUTS", files, [".bib"])
    setup_env_for(type_env, "ENCFONTS", files, [".enc"])
    setup_env_for(
        type_env,
        "LUAINPUTS",
        files,
        [".lua" or ".luc"],
        ":$TEXINPUTS:",
    )
    setup_env_for(type_env, "OPENTYPEFONTS", files)
    setup_env_for(type_env, "T1FONTS", files, [".pfa", ".pfb"])
    setup_env_for(type_env, "TEXFONTMAPS", files, [".map"], ":")
    setup_env_for(type_env, "TEXINPUTS", files)
    setup_env_for(type_env, "TEXPSHEADERS", files, [".pro"])
    setup_env_for(type_env, "TFMFONTS", files, [".tfm"])
    setup_env_for(type_env, "TTFONTS", files, [".ttf", ".ttc"])
    setup_env_for(type_env, "VFFONTS", files, [".vf"])

    env = {
        "MT_MKTEX_OPT": ctx.files.web2c[0].dirname + "/mktex.opt",
        "PATH": ":".join(
            [
                toolchain.kpsewhich.files.to_list()[0].dirname,  # latex bin folder
                toolchain.mktexlsr.files.to_list()[0].dirname,  # script folder
                "/bin",  # sed, rm, etc. needed by mktexlsr
                "/usr/bin",  # needed to find python
                # NOTE: ctx.configuration.default_shell_env returns {}
                # So the default shell env provided by bazel can't
                # be updated by the rules.
                # Supplying the env argument overwrites bazel's env
                # resulting in python not being found.
            ],
        ),
        "SOURCE_DATE_EPOCH": "0",
        "TEXMF": ".",
        "TEXMFCNF": ctx.files.web2c[0].dirname,
        "TEXMFDBS": ".:$TEXMFHOME:$TEXMF",
        "TEXMFHOME": ".",
        "TEXMFROOT": ".",
    }
    env.update(type_env)
    return env

def _latex_impl(ctx):
    toolchain = ctx.toolchains["@bazel_latex//:latex_toolchain_type"].latexinfo

    latex_tool = getattr(toolchain, ctx.attr._engine)
    dep_tools = [
        toolchain.dvisvgm.files,
        toolchain.luatex.files,
        toolchain.gsftopk.files,
        toolchain.kpsewhich.files,
        toolchain.mktexlsr.files,
        toolchain.kpsestat.files,
        toolchain.kpseaccess.files,
    ]

    files = (
        ctx.files._core_dependencies +
        ctx.files.dependencies +
        ctx.files.ini_files +
        ctx.files.font_maps +
        ctx.files.web2c
    )

    fmt = ctx.actions.declare_file(
        ctx.label.name + "/" + ctx.attr._progname + ".fmt",
    )
    ini_args = ctx.actions.args()
    ini_files_path = ctx.files.ini_files[0].dirname
    ini_args.add("-ini")
    ini_args.add(
        "--output-directory",
        fmt.dirname,
    )
    ini_args.add("{}/{}.ini".format(ini_files_path, ctx.attr._progname))

    fontmap_list = []
    for fm in ctx.attr.font_maps:
        for fm_file in fm.files.to_list():
            fontmap_list.append(fm_file.path)

    env = get_env(ctx, toolchain, files)

    ctx.actions.run(
        mnemonic = "LuaLatex",
        executable = latex_tool.files.to_list()[0],
        arguments = [ini_args],
        inputs = depset(
            direct = files,
            transitive = [latex_tool.files] + dep_tools,
        ),
        outputs = [fmt],
        tools = [latex_tool.files.to_list()[0]],
        env = env,
    )

    latex_info = LatexEngineInfo(
        deps = depset(direct = files + [fmt], transitive = [latex_tool.files] + dep_tools),
        toolchain = toolchain,
        fmt = fmt,
        env = env,
        fontmaps = fontmap_list,
    )
    return [latex_info]

latex_engine = rule(
    attrs = {
        "cmd_flags": attr.string_list(
            allow_empty = True,
            default = [],
        ),
        "dependencies": attr.label_list(
            default = [],
        ),
        "font_maps": attr.label_list(
            allow_files = True,
            default = [
                "@texlive_texmf__texmf-dist__fonts__map__dvips__updmap",
                "@texlive_texmf__texmf-dist__fonts__map__pdftex__updmap",
            ],
        ),
        "format": attr.string(
            doc = "Output file format",
            default = "pdf",
            values = ["dvi", "pdf"],
        ),
        "ini_files": attr.label(
            allow_files = True,
            default = "@texlive_texmf__texmf-dist__tex__generic__tex-ini-files",
        ),
        "web2c": attr.label(
            allow_files = True,
            default = "@texlive_texmf__texmf-dist__web2c",
        ),
        "_core_dependencies": attr.label(
            default = "@bazel_latex//:core_dependencies",
        ),
        "_dvi_sub": attr.label(
            default = "@bazel_latex//:dvi_sub",
            executable = True,
            cfg = "exec",
        ),
        # TODO: Suggestion to make _engine public so that the
        #       user can set their engine of choice
        "_engine": attr.string(default = "luahbtex"),
        "_progname": attr.string(default = "lualatex"),
    },
    toolchains = ["@bazel_latex//:latex_toolchain_type"],
    implementation = _latex_impl,
)

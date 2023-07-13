workspace(name = "SnowML")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_jar(
    name = "bazel_diff",
    urls = [
        "https://github.com/Tinder/bazel-diff/releases/download/4.3.0/bazel-diff_deploy.jar",
    ],
    sha256 = "9c4546623a8b9444c06370165ea79a897fcb9881573b18fa5c9ee5c8ba0867e2",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "b8a1527901774180afc798aeb28c4634bdccf19c4d98e7bdd1ce79d1fe9aaad7",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# Latest @ 2023-06-20
# Replace with released version once newer version released.
git_repository(
    name="rules_python",
    commit="0d59fcf561f6d2c4705924bc17c151fb4b998841",
    remote="https://github.com/bazelbuild/rules_python.git"
)

load("//third_party/rules_conda:defs.bzl", "conda_create", "load_conda", "register_toolchain")

http_archive(
    name = "aspect_bazel_lib",
    sha256 = "e3151d87910f69cf1fc88755392d7c878034a69d6499b287bcfc00b1cf9bb415",
    strip_prefix = "bazel-lib-1.32.1",
    url = "https://github.com/aspect-build/bazel-lib/releases/download/v1.32.1/bazel-lib-v1.32.1.tar.gz",
)

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies", "register_yq_toolchains")

aspect_bazel_lib_dependencies()
register_yq_toolchains()

# Below two conda environments (toolchains) are created and they require different
# constraint values. Two platforms defined in bazel/platforms/BUILD provide those
# constraint values. A toolchain matches a platform as long as the platform provides
# all the constraint values the toolchain requires, which means:
# - py3_toolchain_snowflake_conda_only is used iff
#   //bazel/platforms:snowflake_conda_env is the target platform
# - py3_toolchain_extended_channels is used iff
#   //bazel/platforms:extended_conda_env is the target platform
#
# The default platform when --platforms flag is not set, is specified in
# .bazelrc .

load_conda(conda_repo_name = "snowflake_conda", quiet = True)

conda_create(
    name = "py3_env_snowflake_conda_only",
    conda_repo_name = "snowflake_conda",
    timeout = 3600,
    clean = False,
    environment = "@//:conda-env-snowflake.yml",
    coverage_tool = "@//bazel/coverage_tool:coverage_tool.py",
    quiet = True,
)

register_toolchain(
    name = "py3_env_snowflake_conda_only_repo",
    env = "py3_env_snowflake_conda_only",
    target_compatible_with=["@SnowML//bazel/platforms:snowflake_conda_channel"],
    toolchain_name = "py3_toolchain_snowflake_conda_only",
)

load_conda(conda_repo_name = "extended_conda", quiet = True)

conda_create(
    name = "py3_env_extended_channels",
    conda_repo_name = "extended_conda",
    timeout = 3600,
    clean = False,
    environment = "@//:conda-env.yml",
    coverage_tool = "@//bazel/coverage_tool:coverage_tool.py",
    quiet = True,
)

register_toolchain(
    name = "py3_env_extended_channels_repo",
    env = "py3_env_extended_channels",
    target_compatible_with=["@SnowML//bazel/platforms:extended_conda_channels"],
    toolchain_name = "py3_toolchain_extended_channels",
)

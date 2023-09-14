workspace(name = "SnowML")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

http_jar(
    name = "bazel_diff",
    sha256 = "9c4546623a8b9444c06370165ea79a897fcb9881573b18fa5c9ee5c8ba0867e2",
    urls = [
        "https://github.com/Tinder/bazel-diff/releases/download/4.3.0/bazel-diff_deploy.jar",
    ],
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
    name = "rules_python",
    commit = "0d59fcf561f6d2c4705924bc17c151fb4b998841",
    remote = "https://github.com/bazelbuild/rules_python.git",
)

load("//third_party/rules_conda:defs.bzl", "conda_create", "load_conda", "register_toolchain")

http_archive(
    name = "aspect_bazel_lib",
    sha256 = "b44310bef17d33d0e34a624dbbc74de595d37adc16546bd612d6f178eac426e7",
    strip_prefix = "bazel-lib-1.34.2",
    url = "https://github.com/aspect-build/bazel-lib/releases/download/v1.34.2/bazel-lib-v1.34.2.tar.gz",
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

load("@SnowML//bazel/environments:fetch_conda_env_config.bzl", "fetch_conda_env_config")

fetch_conda_env_config(name = "fetch_conda_env_config_repo")

load("@fetch_conda_env_config_repo//:config.bzl", "COMPATIBLE_TARGET", "ENVIRONMENT", "NAME")

load_conda(
    conda_repo_name = "{}_conda".format(NAME),
    quiet = True,
)

conda_create(
    name = "{}_env".format(NAME),
    timeout = 3600,
    clean = False,
    conda_repo_name = "{}_conda".format(NAME),
    coverage_tool = "@//bazel/coverage_tool:coverage_tool.py",
    environment = ENVIRONMENT,
    quiet = True,
)

register_toolchain(
    name = "{}_env_repo".format(NAME),
    env = "{}_env".format(NAME),
    target_compatible_with = COMPATIBLE_TARGET,
    toolchain_name = "py3_toolchain_{}_env".format(NAME),
)

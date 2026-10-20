#!/usr/bin/env bash
set -eux
# The repository cache is not cleaned by `bazel clean --expunge`, so it must live outside
# the conda build prefix; an unset BAZEL_REPOSITORY_CACHE disables it.
BAZEL_EXTRA_BUILD_FLAGS=(--repository_cache="${BAZEL_REPOSITORY_CACHE:-}")
if [[ -n "${BAZEL_DOWNLOADER_CONFIG:-}" ]]; then
    BAZEL_EXTRA_BUILD_FLAGS+=(--experimental_downloader_config="${BAZEL_DOWNLOADER_CONFIG}")
fi
bazel build --config=build "${BAZEL_EXTRA_BUILD_FLAGS[@]}" --config="py${PY_VER}" //:wheel
pip install --no-dependencies "$(bazel info bazel-bin)"/dist/*.whl
# We need to remove all the bazel's output because `conda build` looks at the diff of
# the working environment before and after running this script. And bazel's output
# are included in that working environment.
bazel clean --expunge

# SnowML Conda recipe

Conda's guide on building a conda package from a wheel:
<https://docs.conda.io/projects/conda-build/en/stable/user-guide/wheel-files.html#building-a-conda-package-from-a-wheel-file>

To invoke conda build:

```sh
conda build -c conda-forge --override-channels --prefix-length=0 --python=[3.9|3.10|3.11|3.12] ci/conda_recipe
```

- `--prefix-length=0`: prevent the conda build environment from being created in
   a directory of very long path. conda does that intentionally to make sure
   packages can be installed at peculiar locations. However, it breaks bazel.
   We tried alternatives to make bazel work, but:

  - We don't have a way to cleanly set bazel's output root.
     (<https://github.com/bazelbuild/bazel/issues/4248>).

  - Setting `--output_user_root` causes weird problems (`repository_ctx.execute()`
     any command would result in a `SIGABRT`).

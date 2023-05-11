# Update this for the versions
# Don't change the forth version number from None
VERSION = (0, 2, 4, None)  # this has to be in sync with BUILD.bazel and ci/conda_recipe/meta.yaml


def get_version() -> str:
    """
    Get snowml version.

    Returns:
        snowml version.
    """
    return ".".join([str(d) for d in VERSION if d is not None])

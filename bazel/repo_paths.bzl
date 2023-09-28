"""Helper functions for py_rules.bzl separated for unittesting."""

def _extract_and_split_packages(target, name = None):
    """Splits target into list of packages. For example, '//foo/bar:baz' becomes ['foo', 'bar', 'baz']."""
    colon = target.find(":")
    if colon == -1:
        path = target
    else:
        path = target[:colon]
        name = target[colon:]
    pkgs = path.split("/")
    ret = []
    for p in pkgs:
        if p:
            ret.append(p)
    if name:
        ret.append(name)
    return ret

def _experimental_found(deps):
    """Checks if `experimental` is found in the given list of packages."""
    for d in deps:
        if "experimental" in _extract_and_split_packages(d):
            return True
    return False

def check_for_experimental_dependencies(pkg_name, attrs):
    """Checks if a non-experimental target depends on experimental package. If so, it bails out.

    Args:
      pkg_name(str): Name of a package
      attrs(dict): Attributes dictionary

    Returns:
      True if check passes, False otherwise
    """
    if "deps" not in attrs:
        return True
    if not attrs["deps"]:
        return True
    deps = attrs["deps"]
    experimental_rule = _experimental_found([pkg_name])
    experimental_dependencies = _experimental_found(deps)
    if not experimental_rule and experimental_dependencies:
        return False
    return True

def check_for_tests_dependencies(pkg_name, attrs):
    """Checks if a src target depends on package in tests folder or a target depends on test targets. If so, it bails out.

    Args:
      pkg_name(str): Name of a package
      attrs(dict): Attributes dictionary

    Returns:
      True if check passes, False otherwise
    """
    paths = _extract_and_split_packages(pkg_name, attrs["name"])
    if "deps" not in attrs:
        return True
    if not attrs["deps"]:
        return True
    deps = attrs["deps"]
    for d in deps:
        dep_paths = _extract_and_split_packages(d)
        if len(paths) > 0 and paths[0] == "snowflake":
            if len(dep_paths) > 0 and dep_paths[0] == "tests":
                return False
        elif len(dep_paths[-1]) > 5 and dep_paths[-1][-5:] == "_test":
            return False
    return True

def check_for_test_name(pkg_name, attrs):
    """Checks if a test target have correct name format. If so, it bails out.

    Args:
      pkg_name(str): Name of a package
      attrs(dict): Attributes dictionary

    Returns:
      True if check passes, False otherwise
    """
    paths = _extract_and_split_packages(pkg_name, attrs["name"])
    if len(paths[-1]) >= 5 and paths[-1][-5:] == "_test":
        return True
    return False

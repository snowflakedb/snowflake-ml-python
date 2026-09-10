"""Dependency guardrails for the deprecated ``snowflake.ml.modeling`` estimators.

The ``snowflake.ml.modeling`` estimators are built on ``scikit-learn``, plus ``xgboost`` or
``lightgbm`` for those subpackages. None of these are installed automatically: ``scikit-learn``
and ``xgboost`` became optional in snowflake-ml-python 2.0, while ``lightgbm`` has always been an
optional extra. Every subpackage calls :func:`check_modeling_dependencies` at import time to:

1. Raise an actionable :class:`ImportError` naming only the packages that are actually missing,
   attributing the 2.0 dependency change when it applies to them.
2. Emit a :class:`DeprecationWarning` when the import succeeds, since the estimators are deprecated.

This module must remain importable in environments where the optional packages are absent, so it
never imports ``scikit-learn``, ``xgboost``, or ``lightgbm`` at module load time; availability is
probed with :func:`importlib.util.find_spec`.
"""

import importlib.util
import re
import warnings

# Import name -> the distribution/package name to show the user (for example ``sklearn`` is
# installed as ``scikit-learn``).
_IMPORT_NAME_TO_PACKAGE_NAME = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

# Subpackages of ``snowflake.ml.modeling`` that require an additional optional package beyond
# ``scikit-learn`` (which every modeling estimator depends on). Keyed by the final subpackage name.
_SUBPACKAGE_EXTRA_IMPORT = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

# Packages that stopped being installed automatically in 2.0, so the error message can attribute
# the change to the upgrade. ``lightgbm`` is absent on purpose: it was already an optional extra
# before 2.0, so framing it as a 2.0 change would be misleading.
_NEWLY_OPTIONAL_IN_2_0 = frozenset({"sklearn", "xgboost"})

_MODELING_DEPRECATION_MESSAGE = (
    "snowflake.ml.modeling estimators are deprecated as of snowflake-ml-python 2.0 and will be "
    "removed in a future release. Train models with the native scikit-learn, XGBoost, or LightGBM "
    "estimators and log them to the Snowflake Model Registry instead."
)

# DeprecationWarning is suppressed by default outside of ``__main__``; register a narrowly scoped
# "once" filter so the notice surfaces exactly once per session (mirrors snowflake/ml/model/__init__.py).
warnings.filterwarnings(
    "once",
    message=re.escape(_MODELING_DEPRECATION_MESSAGE),
    category=DeprecationWarning,
)


def _required_import_names(package_name: str) -> list[str]:
    """Return the import names required by a ``snowflake.ml.modeling`` subpackage."""
    subpackage = package_name.rsplit(".", 1)[-1]
    required = ["sklearn"]
    extra_import = _SUBPACKAGE_EXTRA_IMPORT.get(subpackage)
    if extra_import is not None:
        required.append(extra_import)
    return required


def _is_installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ModuleNotFoundError, ValueError):
        # ``find_spec`` raises ``ModuleNotFoundError`` when a parent package is missing; treat any
        # such failure as "not installed" so we surface the actionable error below.
        return False


def _format_package_list(package_names: list[str]) -> str:
    return " and ".join(package_names)


def _optional_dependency_clause(missing_import_names: list[str], package_list: str, *, plural: bool) -> str:
    """Explain why the missing packages are absent, attributing the 2.0 change only where it applies."""
    verb = "are" if plural else "is"
    noun = "optional dependencies" if plural else "an optional dependency"
    newly_optional = sorted(
        _IMPORT_NAME_TO_PACKAGE_NAME[name] for name in missing_import_names if name in _NEWLY_OPTIONAL_IN_2_0
    )
    if not newly_optional:
        return f"{package_list} {verb} {noun} of snowflake-ml-python that {verb} not installed automatically."
    if len(newly_optional) == len(missing_import_names):
        return (
            f"As of snowflake-ml-python 2.0, {package_list} {verb} {noun} of snowflake-ml-python "
            f"that {verb} no longer installed automatically."
        )
    return (
        f"{package_list} {verb} {noun} of snowflake-ml-python that {verb} not installed "
        f"automatically ({_format_package_list(newly_optional)} as of snowflake-ml-python 2.0)."
    )


def _missing_dependency_error_message(package_name: str, missing_import_names: list[str]) -> str:
    package_names = sorted(_IMPORT_NAME_TO_PACKAGE_NAME[name] for name in missing_import_names)
    package_list = _format_package_list(package_names)
    plural = len(package_names) > 1
    verb = "are" if plural else "is"
    install_noun = "packages" if plural else "package"
    optional_clause = _optional_dependency_clause(missing_import_names, package_list, plural=plural)
    return (
        f"'{package_name}' requires {package_list}, which {verb} not installed. "
        f"{optional_clause} Install the missing {install_noun} in your environment to continue "
        "using snowflake.ml.modeling."
    )


def check_modeling_dependencies(package_name: str, *, emit_deprecation_warning: bool = True) -> None:
    """Verify optional dependencies for a ``snowflake.ml.modeling`` subpackage and warn on deprecation.

    Args:
        package_name: The fully qualified name of the importing subpackage (pass ``__name__``), for
            example ``"snowflake.ml.modeling.linear_model"``. Determines which optional packages are
            required: always ``scikit-learn``, plus ``xgboost``/``lightgbm`` for those subpackages.
        emit_deprecation_warning: When True (the default) and all required dependencies are present,
            emit a :class:`DeprecationWarning` that ``snowflake.ml.modeling`` estimators are
            deprecated. Set False for non-estimator subpackages (for example ``metrics``).

    Raises:
        ImportError: If a required optional package is not installed, with instructions on how to
            install it.
    """
    missing = [name for name in _required_import_names(package_name) if not _is_installed(name)]
    if missing:
        raise ImportError(_missing_dependency_error_message(package_name, missing))

    if emit_deprecation_warning:
        warnings.warn(_MODELING_DEPRECATION_MESSAGE, category=DeprecationWarning, stacklevel=2)

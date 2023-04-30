from typing import List, Optional, Tuple

from conda import exceptions as conda_exceptions
from conda_libmamba_solver import solver


def _resolve_dependencies(packages: List[str], channels: List[str]) -> Optional[List[Tuple[str, str]]]:
    """Use conda api to check if given packages are resolvable in given channels.

    Args:
        packages: Packages to be installed.
        channels: Anaconda channels (name or url) where conda should search into.

    Returns:
        List of frozen dependencies represented in tuple (name, version) if resolvable, None otherwise.
    """
    conda_solver = solver.LibMambaSolver("snow-env", channels=channels, specs_to_add=packages)
    try:
        solve_result = conda_solver.solve_final_state()
    except (
        conda_exceptions.ResolvePackageNotFound,
        conda_exceptions.UnsatisfiableError,
        conda_exceptions.PackagesNotFoundError,
        solver.LibMambaUnsatisfiableError,
    ):
        return None

    return [(pkg_record.name, pkg_record.version) for pkg_record in solve_result]

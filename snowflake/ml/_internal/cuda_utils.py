from dataclasses import dataclass
from typing import List, Optional

from packaging import version


# TODO(halu): Once we add multiple py version support
# we should include as well as py3.8 is gradually
# on its path to be dropped.
@dataclass(frozen=True)
class TorchCompatibilityConfig:
    # Pytorch version(major.minor.micro).
    torch: str
    # List of supporting CUDA versions(major.minor).
    cudas: List[str]


# Used for testing only to make sure valid config
# Make sure check with SPCS team before bumping this up.
_SPCS_CUDA_VERSION = "12.1"

# Drop support for 1.*
_TORCH_CUDA_COMPAT_CONFIGS = [
    TorchCompatibilityConfig(torch="2.1.0", cudas=["11.8", "12.1"]),
    TorchCompatibilityConfig(torch="2.0.1", cudas=["11.7", "11.8"]),
    TorchCompatibilityConfig(torch="2.0.0", cudas=["11.7", "11.8"]),
]


def _normalized_cuda_version(user_cuda_version: str) -> str:
    """Normalize cuda version to major.minor only.
    We round down the micro version to rely on backward compatibility
    Rather than forward.

    Args:
        user_cuda_version: User local or provided full cuda version.

    Returns:
        Normalized version
    """
    v = version.Version(user_cuda_version)
    return f"{v.major}.{v.minor}"


def _normalized_torch_version(user_torch_version: str) -> str:
    """Normalize torch version to release version.

    Args:
        user_torch_version: User local or provided full torch version.

    Returns:
        Normalized "major.minor.micro".

    Raises:
        InvalidVersion if not PEP 440 compliant.
    """
    v = version.Version(user_torch_version)
    return f"{v.major}.{v.minor}.{v.micro}"


def is_torch_cuda_compatible(
    torch_version: str,
    cuda_version: str,
) -> bool:
    """Check if provided pair is compatible.

    Args:
        torch_version: User torch version.
        cuda_version: User cuda version.

    Returns:
        True if compatible.
    """
    n_torch_version = _normalized_torch_version(torch_version)
    n_cuda = _normalized_cuda_version(cuda_version)
    for cfg in _TORCH_CUDA_COMPAT_CONFIGS:
        if cfg.torch == n_torch_version:
            if n_cuda in cfg.cudas:
                return True
            else:
                return False
    return False


def get_latest_cuda_for_torch(torch_version: str) -> Optional[str]:
    """Get latest supporting CUDA version if possible.

    Args:
        torch_version (str): User torch version.

    Returns:
        Latest supporting CUDA version or None.
    """
    parsed_torch_version = _normalized_torch_version(torch_version)
    for cfg in _TORCH_CUDA_COMPAT_CONFIGS:
        if cfg.torch == parsed_torch_version:
            return sorted(cfg.cudas, reverse=True)[0]
    return None

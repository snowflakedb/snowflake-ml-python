from enum import Enum


class RelaxVersionStrategy(Enum):
    NO_RELAX = "no_relax"
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "major"


RELAX_VERSION_STRATEGY_MAP = {
    # The version of cloudpickle should not be relaxed as it is used for serialization.
    "cloudpickle": RelaxVersionStrategy.NO_RELAX,
    # The version of scikit-learn should be relaxed only in patch version as it has breaking changes in minor version.
    "scikit-learn": RelaxVersionStrategy.PATCH,
}

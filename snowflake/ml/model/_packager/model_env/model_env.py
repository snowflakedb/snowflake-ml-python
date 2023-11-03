import collections
import itertools
import os
import pathlib
import warnings
from typing import DefaultDict, List, Optional

from packaging import requirements, version

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model._packager.model_meta import model_meta_schema

# requirement: Full version requirement where name is conda package name.
# pypi_name: The name of dependency in Pypi.
ModelDependency = collections.namedtuple("ModelDependency", ["requirement", "pip_name"])

_DEFAULT_ENV_DIR = "env"
_DEFAULT_CONDA_ENV_FILENAME = "conda.yml"
_DEFAULT_PIP_REQUIREMENTS_FILENAME = "requirements.txt"

# The default CUDA version is chosen based on the driver availability in SPCS.
# If changing this version, we need also change the version of default PyTorch in HuggingFace pipeline handler to
# make sure they are compatible.
DEFAULT_CUDA_VERSION = "11.7"


class ModelEnv:
    def __init__(
        self,
        conda_env_rel_path: Optional[str] = None,
        pip_requirements_rel_path: Optional[str] = None,
    ) -> None:
        if conda_env_rel_path is None:
            conda_env_rel_path = os.path.join(_DEFAULT_ENV_DIR, _DEFAULT_CONDA_ENV_FILENAME)
        if pip_requirements_rel_path is None:
            pip_requirements_rel_path = os.path.join(_DEFAULT_ENV_DIR, _DEFAULT_PIP_REQUIREMENTS_FILENAME)
        self.conda_env_rel_path = pathlib.PurePosixPath(pathlib.Path(conda_env_rel_path).as_posix())
        self.pip_requirements_rel_path = pathlib.PurePosixPath(pathlib.Path(pip_requirements_rel_path).as_posix())
        self._conda_dependencies: DefaultDict[str, List[requirements.Requirement]] = collections.defaultdict(list)
        self._pip_requirements: List[requirements.Requirement] = []
        self._python_version: version.Version = version.parse(snowml_env.PYTHON_VERSION)
        self._cuda_version: Optional[version.Version] = None
        self._snowpark_ml_version: version.Version = version.parse(snowml_env.VERSION)

    @property
    def conda_dependencies(self) -> List[str]:
        """List of conda channel and dependencies from that to run the model"""
        return sorted(
            f"{chan}::{str(req)}" if chan else str(req)
            for chan, reqs in self._conda_dependencies.items()
            for req in reqs
        )

    @conda_dependencies.setter
    def conda_dependencies(
        self,
        conda_dependencies: Optional[List[str]] = None,
    ) -> None:
        self._conda_dependencies = env_utils.validate_conda_dependency_string_list(
            conda_dependencies if conda_dependencies else []
        )

    @property
    def pip_requirements(self) -> List[str]:
        """List of pip Python packages requirements for running the model."""
        return sorted(list(map(str, self._pip_requirements)))

    @pip_requirements.setter
    def pip_requirements(
        self,
        pip_requirements: Optional[List[str]] = None,
    ) -> None:
        self._pip_requirements = env_utils.validate_pip_requirement_string_list(
            pip_requirements if pip_requirements else []
        )

    @property
    def python_version(self) -> str:
        return f"{self._python_version.major}.{self._python_version.minor}"

    @python_version.setter
    def python_version(self, python_version: Optional[str] = None) -> None:
        if python_version:
            self._python_version = version.parse(python_version)

    @property
    def cuda_version(self) -> Optional[str]:
        if self._cuda_version:
            return f"{self._cuda_version.major}.{self._cuda_version.minor}"
        return None

    @cuda_version.setter
    def cuda_version(self, cuda_version: Optional[str] = None) -> None:
        # We need to check this as CUDA version would be set inside the handler, while python_version or snowpark
        # ML version would not.
        if cuda_version:
            parsed_cuda_version = version.parse(cuda_version)
            if self._cuda_version is None:
                self._cuda_version = parsed_cuda_version
            else:
                if self.cuda_version != f"{parsed_cuda_version.major}.{parsed_cuda_version.minor}":
                    raise ValueError(
                        f"Different CUDA version {self.cuda_version} and {cuda_version} found in the same model!"
                    )

    @property
    def snowpark_ml_version(self) -> str:
        return str(self._snowpark_ml_version)

    @snowpark_ml_version.setter
    def snowpark_ml_version(self, snowpark_ml_version: Optional[str] = None) -> None:
        if snowpark_ml_version:
            self._snowpark_ml_version = version.parse(snowpark_ml_version)

    def include_if_absent(self, pkgs: List[ModelDependency], check_local_version: bool = False) -> None:
        """Append requirements into model env if absent.

        Args:
            pkgs: A list of ModelDependency namedtuple to be appended.
            check_local_version: Flag to indicate if it is required to pin to local version. Defaults to False.
        """
        conda_reqs_str, pip_names_str = tuple(zip(*pkgs))
        pip_names = env_utils.validate_pip_requirement_string_list(list(pip_names_str))
        conda_reqs = env_utils.validate_conda_dependency_string_list(list(conda_reqs_str))

        for conda_req, pip_name in zip(conda_reqs[env_utils.DEFAULT_CHANNEL_NAME], pip_names):
            if check_local_version:
                req_to_check = requirements.Requirement(f"{pip_name.name}{conda_req.specifier}")
                req_to_add = env_utils.get_local_installed_version_of_pip_package(req_to_check)
            else:
                req_to_add = conda_req
            added_in_pip = False
            for added_pip_req in self._pip_requirements:
                if added_pip_req.name == pip_name.name:
                    warnings.warn(
                        (
                            f"Basic dependency {req_to_add.name} specified from PIP requirements."
                            + " This may prevent model deploying to Snowflake Warehouse."
                        ),
                        category=UserWarning,
                        stacklevel=2,
                    )
                    added_in_pip = True
            if added_in_pip:
                continue
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies, (env_utils.DEFAULT_CHANNEL_NAME, req_to_add)
                )
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                warnings.warn(
                    (
                        f"Basic dependency {req_to_add.name} specified from non-Snowflake channel."
                        + " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
                    stacklevel=2,
                )

    def generate_env_for_cuda(self) -> None:
        if self.cuda_version is None:
            return

        cuda_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="cuda", remove_spec=False
        )
        if cuda_spec and not cuda_spec.specifier.contains(self.cuda_version):
            raise ValueError(
                "The CUDA requirement you specified in your conda dependencies or pip requirements is"
                " conflicting with CUDA version required. Please do not specify CUDA dependency using conda"
                " dependencies or pip requirements."
            )

        if not cuda_spec:
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    ("nvidia", requirements.Requirement(f"cuda=={self.cuda_version}.*")),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass

        xgboost_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="xgboost", remove_spec=True
        )
        if xgboost_spec:
            xgboost_spec.name = "py-xgboost-gpu"
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    ("conda-forge", xgboost_spec),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass

        pytorch_spec = env_utils.find_dep_spec(
            self._conda_dependencies,
            self._pip_requirements,
            conda_pkg_name="pytorch",
            pip_pkg_name="torch",
            remove_spec=True,
        )
        pytorch_cuda_spec = env_utils.find_dep_spec(
            self._conda_dependencies,
            self._pip_requirements,
            conda_pkg_name="pytorch-cuda",
            remove_spec=False,
        )
        if pytorch_cuda_spec and not pytorch_cuda_spec.specifier.contains(self.cuda_version):
            raise ValueError(
                "The Pytorch-CUDA requirement you specified in your conda dependencies or pip requirements is"
                " conflicting with CUDA version required. Please do not specify Pytorch-CUDA dependency using conda"
                " dependencies or pip requirements."
            )
        if pytorch_spec:
            pytorch_spec.name = "pytorch"
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    ("pytorch", pytorch_spec),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass
            if not pytorch_cuda_spec:
                try:
                    env_utils.append_conda_dependency(
                        self._conda_dependencies,
                        p_chan_dep=("pytorch", requirements.Requirement(f"pytorch-cuda=={self.cuda_version}.*")),
                    )
                except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                    pass

        tf_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="tensorflow", remove_spec=True
        )
        if tf_spec:
            tf_spec.name = "tensorflow-gpu"
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    ("conda-forge", tf_spec),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass

        transformers_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="transformers", remove_spec=False
        )
        if transformers_spec:
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    ("conda-forge", requirements.Requirement("accelerate>=0.22.0")),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass

            # Required by bitsandbytes
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies,
                    (
                        env_utils.DEFAULT_CHANNEL_NAME,
                        env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("scipy")),
                    ),
                )
            except (env_utils.DuplicateDependencyError, env_utils.DuplicateDependencyInMultipleChannelsError):
                pass

            try:
                env_utils.append_requirement_list(
                    self._pip_requirements,
                    requirements.Requirement("bitsandbytes>=0.41.0"),
                )
            except env_utils.DuplicateDependencyError:
                pass

    def relax_version(self) -> None:
        """Relax the version requirements for both conda dependencies and pip requirements.
        It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1)
        """
        self._conda_dependencies = collections.defaultdict(
            list,
            {
                chan: list(map(env_utils.relax_requirement_version, deps))
                for chan, deps in self._conda_dependencies.items()
            },
        )
        self._pip_requirements = list(map(env_utils.relax_requirement_version, self._pip_requirements))

    def load_from_conda_file(self, conda_env_path: pathlib.Path) -> None:
        conda_dependencies_dict, pip_requirements_list, python_version = env_utils.load_conda_env_file(conda_env_path)

        for channel, channel_dependencies in conda_dependencies_dict.items():
            if channel != env_utils.DEFAULT_CHANNEL_NAME:
                warnings.warn(
                    (
                        "Found dependencies specified in the conda file from non-Snowflake channel."
                        " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
                )
            if len(channel_dependencies) == 0 and channel not in self._conda_dependencies:
                warnings.warn(
                    (
                        f"Found additional conda channel {channel} specified in the conda file."
                        " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
                )
                self._conda_dependencies[channel] = []

            for channel_dependency in channel_dependencies:
                try:
                    env_utils.append_conda_dependency(self._conda_dependencies, (channel, channel_dependency))
                except env_utils.DuplicateDependencyError:
                    pass
                except env_utils.DuplicateDependencyInMultipleChannelsError:
                    warnings.warn(
                        (
                            f"Dependency {channel_dependency.name} appeared in multiple channels as conda dependency."
                            " This may be unintentional."
                        ),
                        category=UserWarning,
                    )

        if pip_requirements_list:
            warnings.warn(
                (
                    "Found dependencies specified as pip requirements."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
                category=UserWarning,
            )
            for pip_dependency in pip_requirements_list:
                if any(
                    channel_dependency.name == pip_dependency.name
                    for channel_dependency in itertools.chain(*self._conda_dependencies.values())
                ):
                    continue
                env_utils.append_requirement_list(self._pip_requirements, pip_dependency)

        if python_version:
            self.python_version = python_version

    def load_from_pip_file(self, pip_requirements_path: pathlib.Path) -> None:
        pip_requirements_list = env_utils.load_requirements_file(pip_requirements_path)

        if pip_requirements_list:
            warnings.warn(
                (
                    "Found dependencies specified as pip requirements."
                    " This may prevent model deploying to Snowflake Warehouse."
                ),
                category=UserWarning,
            )
            for pip_dependency in pip_requirements_list:
                if any(
                    channel_dependency.name == pip_dependency.name
                    for channel_dependency in itertools.chain(*self._conda_dependencies.values())
                ):
                    continue
                env_utils.append_requirement_list(self._pip_requirements, pip_dependency)

    def load_from_dict(self, base_dir: pathlib.Path, env_dict: model_meta_schema.ModelEnvDict) -> None:
        self.conda_env_rel_path = pathlib.PurePosixPath(env_dict["conda"])
        self.pip_requirements_rel_path = pathlib.PurePosixPath(env_dict["pip"])

        self.load_from_conda_file(base_dir / self.conda_env_rel_path)
        self.load_from_pip_file(base_dir / self.pip_requirements_rel_path)

        self.python_version = env_dict["python_version"]
        self.cuda_version = env_dict.get("cuda_version", None)
        self.snowpark_ml_version = env_dict["snowpark_ml_version"]

    def save_as_dict(self, base_dir: pathlib.Path) -> model_meta_schema.ModelEnvDict:
        env_utils.save_conda_env_file(
            pathlib.Path(base_dir / self.conda_env_rel_path), self._conda_dependencies, self.python_version
        )
        env_utils.save_requirements_file(
            pathlib.Path(base_dir / self.pip_requirements_rel_path), self._pip_requirements
        )
        return {
            "conda": self.conda_env_rel_path.as_posix(),
            "pip": self.pip_requirements_rel_path.as_posix(),
            "python_version": self.python_version,
            "cuda_version": self.cuda_version,
            "snowpark_ml_version": self.snowpark_ml_version,
        }
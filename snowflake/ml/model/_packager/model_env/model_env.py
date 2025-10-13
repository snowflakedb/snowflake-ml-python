import collections
import itertools
import os
import pathlib
import warnings
from typing import DefaultDict, Optional

from packaging import requirements, version

from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager.model_meta import model_meta_schema

# requirement: Full version requirement where name is conda package name.
# pypi_name: The name of dependency in Pypi.
ModelDependency = collections.namedtuple("ModelDependency", ["requirement", "pip_name"])

_DEFAULT_ENV_DIR = "env"
_DEFAULT_CONDA_ENV_FILENAME = "conda.yml"
_DEFAULT_PIP_REQUIREMENTS_FILENAME = "requirements.txt"

# The default CUDA version is chosen based on the driver availability in SPCS.
# Make sure they are aligned with default CUDA version in inference server.
DEFAULT_CUDA_VERSION = "12.4"


class ModelEnv:
    def __init__(
        self,
        conda_env_rel_path: Optional[str] = None,
        pip_requirements_rel_path: Optional[str] = None,
        prefer_pip: bool = False,
        target_platforms: Optional[list[model_types.TargetPlatform]] = None,
    ) -> None:
        if conda_env_rel_path is None:
            conda_env_rel_path = os.path.join(_DEFAULT_ENV_DIR, _DEFAULT_CONDA_ENV_FILENAME)
        if pip_requirements_rel_path is None:
            pip_requirements_rel_path = os.path.join(_DEFAULT_ENV_DIR, _DEFAULT_PIP_REQUIREMENTS_FILENAME)
        self.prefer_pip: bool = prefer_pip
        self.conda_env_rel_path = pathlib.PurePosixPath(pathlib.Path(conda_env_rel_path).as_posix())
        self.pip_requirements_rel_path = pathlib.PurePosixPath(pathlib.Path(pip_requirements_rel_path).as_posix())
        self.artifact_repository_map: Optional[dict[str, str]] = None
        self.resource_constraint: Optional[dict[str, str]] = None
        self._conda_dependencies: DefaultDict[str, list[requirements.Requirement]] = collections.defaultdict(list)
        self._pip_requirements: list[requirements.Requirement] = []
        self._python_version: version.Version = version.parse(snowml_env.PYTHON_VERSION)
        self._cuda_version: Optional[version.Version] = None
        self._snowpark_ml_version: version.Version = version.parse(snowml_version.VERSION)
        self._target_platforms = target_platforms
        self._warnings_shown: set[str] = set()

    @property
    def conda_dependencies(self) -> list[str]:
        """List of conda channel and dependencies from that to run the model"""
        return sorted(
            f"{chan}::{str(req)}" if chan else str(req)
            for chan, reqs in self._conda_dependencies.items()
            for req in reqs
        )

    @conda_dependencies.setter
    def conda_dependencies(
        self,
        conda_dependencies: Optional[list[str]] = None,
    ) -> None:
        self._conda_dependencies = env_utils.validate_conda_dependency_string_list(
            conda_dependencies if conda_dependencies else [], add_local_version_specifier=True
        )

    @property
    def pip_requirements(self) -> list[str]:
        """List of pip Python packages requirements for running the model."""
        return sorted(list(map(str, self._pip_requirements)))

    @pip_requirements.setter
    def pip_requirements(
        self,
        pip_requirements: Optional[list[str]] = None,
    ) -> None:
        self._pip_requirements = env_utils.validate_pip_requirement_string_list(
            pip_requirements if pip_requirements else [], add_local_version_specifier=True
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

    @property
    def targets_warehouse(self) -> bool:
        """Returns True if warehouse is a target platform."""
        return self._target_platforms is None or model_types.TargetPlatform.WAREHOUSE in self._target_platforms

    def _warn_once(self, message: str, stacklevel: int = 2) -> None:
        """Show warning only once per ModelEnv instance."""
        if message not in self._warnings_shown:
            warnings.warn(message, category=UserWarning, stacklevel=stacklevel)
            self._warnings_shown.add(message)

    def include_if_absent(
        self,
        pkgs: list[ModelDependency],
        check_local_version: bool = False,
    ) -> None:
        """Append requirements into model env if absent. Depending on the environment, requirements may be added
        to either the pip requirements or conda dependencies.

        Args:
            pkgs: A list of ModelDependency namedtuple to be appended.
            check_local_version: Flag to indicate if it is required to pin to local version. Defaults to False.
        """
        if (self.pip_requirements or self.prefer_pip) and not self.conda_dependencies and pkgs:
            pip_pkg_reqs: list[str] = []
            if self.targets_warehouse and not self.artifact_repository_map:
                self._warn_once(
                    (
                        "Dependencies specified from pip requirements."
                        " This may prevent model deploying to Snowflake Warehouse."
                        " Use 'artifact_repository_map' to deploy the model to Warehouse."
                    ),
                    stacklevel=2,
                )
            for conda_req_str, pip_name in pkgs:
                _, conda_req = env_utils._validate_conda_dependency_string(conda_req_str)
                pip_req = requirements.Requirement(f"{pip_name}{conda_req.specifier}")
                pip_pkg_reqs.append(str(pip_req))
            self._include_if_absent_pip(pip_pkg_reqs, check_local_version)
        else:
            self._include_if_absent_conda(pkgs, check_local_version)

    def _include_if_absent_conda(self, pkgs: list[ModelDependency], check_local_version: bool = False) -> None:
        """Append requirements into model env conda dependencies if absent.

        Args:
            pkgs: A list of ModelDependency namedtuple to be appended.
            check_local_version: Flag to indicate if it is required to pin to local version. Defaults to False.
        """

        for conda_req_str, pip_name in pkgs:
            conda_req_channel, conda_req = env_utils._validate_conda_dependency_string(conda_req_str)
            if check_local_version:
                req_to_check = requirements.Requirement(f"{pip_name}{conda_req.specifier}")
                req_to_add = env_utils.get_local_installed_version_of_pip_package(req_to_check)
                req_to_add.name = conda_req.name
            else:
                req_to_add = conda_req
            show_warning_message = (
                conda_req_channel == env_utils.DEFAULT_CHANNEL_NAME
                and self.targets_warehouse
                and not self.artifact_repository_map
            )

            if any(added_pip_req.name == pip_name for added_pip_req in self._pip_requirements):
                if show_warning_message:
                    self._warn_once(
                        (
                            f"Basic dependency {req_to_add.name} specified from pip requirements."
                            " This may prevent model deploying to Snowflake Warehouse."
                            " Use 'artifact_repository_map' to deploy the model to Warehouse."
                        ),
                        stacklevel=2,
                    )
                continue

            try:
                env_utils.append_conda_dependency(self._conda_dependencies, (conda_req_channel, req_to_add))
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                if show_warning_message:
                    self._warn_once(
                        (
                            f"Basic dependency {req_to_add.name} specified from non-Snowflake channel."
                            + " This may prevent model deploying to Snowflake Warehouse."
                        ),
                        stacklevel=2,
                    )

    def _include_if_absent_pip(self, pkgs: list[str], check_local_version: bool = False) -> None:
        """Append pip requirements into model env pip requirements if absent.

        Args:
            pkgs: A list of strings to be appended to pip environment.
            check_local_version: Flag to indicate if it is required to pin to local version. Defaults to False.
        """

        pip_reqs = env_utils.validate_pip_requirement_string_list(pkgs)
        for pip_req in pip_reqs:
            if check_local_version:
                pip_req = env_utils.get_local_installed_version_of_pip_package(pip_req)
            try:
                env_utils.append_requirement_list(self._pip_requirements, pip_req)
            except env_utils.DuplicateDependencyError:
                pass

    def remove_if_present_conda(self, conda_pkgs: list[str]) -> None:
        """Remove conda requirements from model env if present.

        Args:
            conda_pkgs: A list of package name to be removed from conda requirements.
        """
        for pkg_name in conda_pkgs:
            spec_conda = env_utils._find_conda_dep_spec(self._conda_dependencies, pkg_name)
            if spec_conda:
                channel, spec = spec_conda
                self._conda_dependencies[channel].remove(spec)

    def generate_env_for_cuda(self) -> None:

        # Insert py-xgboost-gpu only for XGBoost versions < 3.0.0
        xgboost_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="xgboost", remove_spec=False
        )
        if xgboost_spec:
            # Only handle explicitly pinned versions. Insert GPU variant iff pinned major < 3.
            pinned_major: Optional[int] = None
            for spec in xgboost_spec.specifier:
                if spec.operator in ("==", "===", ">", ">="):
                    try:
                        pinned_major = version.parse(spec.version).major
                    except version.InvalidVersion:
                        pinned_major = None
                    break

            if pinned_major is not None and pinned_major < 3:
                xgboost_spec = env_utils.find_dep_spec(
                    self._conda_dependencies, self._pip_requirements, conda_pkg_name="xgboost", remove_spec=True
                )
                if xgboost_spec:
                    self.include_if_absent(
                        [ModelDependency(requirement=f"py-xgboost-gpu{xgboost_spec.specifier}", pip_name="xgboost")],
                        check_local_version=False,
                    )

        tf_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="tensorflow", remove_spec=True
        )
        if tf_spec:
            self.include_if_absent(
                [ModelDependency(requirement=f"tensorflow-gpu{tf_spec.specifier}", pip_name="tensorflow")],
                check_local_version=False,
            )

        transformers_spec = env_utils.find_dep_spec(
            self._conda_dependencies, self._pip_requirements, conda_pkg_name="transformers", remove_spec=False
        )
        if transformers_spec:
            self.include_if_absent(
                [
                    ModelDependency(requirement="accelerate>=0.22.0", pip_name="accelerate"),
                    ModelDependency(requirement="scipy>=1.9", pip_name="scipy"),
                ],
                check_local_version=False,
            )

            self._include_if_absent_pip(["bitsandbytes>=0.41.0"], check_local_version=False)

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
        conda_dependencies_dict, pip_requirements_list, python_version, cuda_version = env_utils.load_conda_env_file(
            conda_env_path
        )

        for channel, channel_dependencies in conda_dependencies_dict.items():
            if channel != env_utils.DEFAULT_CHANNEL_NAME and self.targets_warehouse:
                self._warn_once(
                    (
                        "Found dependencies specified in the conda file from non-Snowflake channel."
                        " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    stacklevel=2,
                )
            if len(channel_dependencies) == 0 and channel not in self._conda_dependencies and self.targets_warehouse:
                self._warn_once(
                    (
                        f"Found additional conda channel {channel} specified in the conda file."
                        " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    stacklevel=2,
                )
                self._conda_dependencies[channel] = []

            for channel_dependency in channel_dependencies:
                try:
                    env_utils.append_conda_dependency(self._conda_dependencies, (channel, channel_dependency))
                except env_utils.DuplicateDependencyError:
                    pass
                except env_utils.DuplicateDependencyInMultipleChannelsError:
                    self._warn_once(
                        (
                            f"Dependency {channel_dependency.name} appeared in multiple channels as conda dependency."
                            " This may be unintentional."
                        ),
                        stacklevel=2,
                    )

        if pip_requirements_list and self.targets_warehouse:
            if not self.artifact_repository_map:
                self._warn_once(
                    (
                        "Found dependencies specified as pip requirements."
                        " This may prevent model deploying to Snowflake Warehouse."
                        " Use 'artifact_repository_map' to deploy the model to Warehouse."
                    ),
                    stacklevel=2,
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

        if cuda_version:
            self.cuda_version = cuda_version

    def load_from_pip_file(self, pip_requirements_path: pathlib.Path) -> None:
        pip_requirements_list = env_utils.load_requirements_file(pip_requirements_path)

        if pip_requirements_list and self.targets_warehouse:
            if not self.artifact_repository_map:
                self._warn_once(
                    (
                        "Found dependencies specified as pip requirements."
                        " This may prevent model deploying to Snowflake Warehouse."
                        " Use 'artifact_repository_map' to deploy the model to Warehouse."
                    ),
                    stacklevel=2,
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
        self.artifact_repository_map = env_dict.get("artifact_repository_map")
        self.resource_constraint = env_dict.get("resource_constraint")

        self.load_from_conda_file(base_dir / self.conda_env_rel_path)
        self.load_from_pip_file(base_dir / self.pip_requirements_rel_path)

        self.python_version = env_dict["python_version"]
        self.cuda_version = env_dict.get("cuda_version")
        self.snowpark_ml_version = env_dict["snowpark_ml_version"]

    def save_as_dict(
        self,
        base_dir: pathlib.Path,
        default_channel_override: str = env_utils.SNOWFLAKE_CONDA_CHANNEL_URL,
        is_gpu: Optional[bool] = False,
    ) -> model_meta_schema.ModelEnvDict:
        cuda_version = self.cuda_version if is_gpu else None
        env_utils.save_conda_env_file(
            pathlib.Path(base_dir / self.conda_env_rel_path),
            self._conda_dependencies,
            self.python_version,
            cuda_version,
            default_channel_override=default_channel_override,
        )
        env_utils.save_requirements_file(
            pathlib.Path(base_dir / self.pip_requirements_rel_path), self._pip_requirements
        )
        return {
            "conda": self.conda_env_rel_path.as_posix(),
            "pip": self.pip_requirements_rel_path.as_posix(),
            "artifact_repository_map": self.artifact_repository_map or {},
            "resource_constraint": self.resource_constraint or {},
            "python_version": self.python_version,
            "cuda_version": self.cuda_version,
            "snowpark_ml_version": self.snowpark_ml_version,
        }

    def validate_with_local_env(
        self, check_snowpark_ml_version: bool = False
    ) -> list[env_utils.IncorrectLocalEnvironmentError]:
        errors = []
        try:
            env_utils.validate_py_runtime_version(str(self._python_version))
        except env_utils.IncorrectLocalEnvironmentError as e:
            errors.append(e)

        for conda_reqs in self._conda_dependencies.values():
            for conda_req in conda_reqs:
                try:
                    env_utils.validate_local_installed_version_of_pip_package(
                        env_utils.try_convert_conda_requirement_to_pip(conda_req)
                    )
                except env_utils.IncorrectLocalEnvironmentError as e:
                    errors.append(e)

        for pip_req in self._pip_requirements:
            try:
                env_utils.validate_local_installed_version_of_pip_package(pip_req)
            except env_utils.IncorrectLocalEnvironmentError as e:
                errors.append(e)

        if check_snowpark_ml_version:
            # For Modeling model
            if self._snowpark_ml_version.base_version != snowml_version.VERSION:
                errors.append(
                    env_utils.IncorrectLocalEnvironmentError(
                        f"The local installed version of Snowpark ML library is {snowml_version.VERSION} "
                        f"which differs from required version {self.snowpark_ml_version}."
                    )
                )

        return errors

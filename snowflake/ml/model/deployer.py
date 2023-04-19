from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from snowflake.ml.model import _udf_util, model
from snowflake.snowpark import DataFrame, Session, functions as F
from snowflake.snowpark._internal import type_utils


class TargetPlatform(Enum):
    SNOWPARK = "snowpark"
    WAREHOUSE = "warehouse"


@dataclass
class Deployment:
    """Deployment information.

    Attributes:
        name: Name of the deployment.
        model_dir_path: Local path to the directory of the packed model.
        platform: Target platform to deploy the model.
        options: Additional options when deploying the model.
    """

    name: str
    model_dir_path: str
    platform: TargetPlatform
    options: Dict[str, str]


class DeploymentManager(ABC):
    """WIP: Intended to provide model deployment management.
    Abstract class for a deployment manager.
    """

    @abstractmethod
    def create(self, name: str, model_dir_path: str, platform: TargetPlatform, options: Dict[str, str]) -> Deployment:
        """Create a deployment.

        Args:
            name: Name of the deployment for the model.
            model_dir_path: Directory of the model.
            platform: Target platform to deploy the model.
            options: Additional options when deploying the model.
                Each target platform will have their own specifications of options.
        """
        pass

    @abstractmethod
    def list(self) -> List[Deployment]:
        """List all deployment in this manager."""
        pass

    @abstractmethod
    def get(self, name: str) -> Optional[Deployment]:
        """Get a specific deployment with the given name in this manager.

        Args:
            name: Name of deployment.
        """
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete a deployment with the given name in this manager.

        Args:
            name: Name of deployment.
        """
        pass


class LocalDeploymentManager(DeploymentManager):
    """A simplest implementation of Deployment Manager that store the deployment information locally."""

    def __init__(self) -> None:
        self._storage: Dict[str, Deployment] = dict()

    def create(self, name: str, model_dir_path: str, platform: TargetPlatform, options: Dict[str, Any]) -> Deployment:
        """Create a deployment.

        Args:
            name: Name of the deployment for the model.
            model_dir_path: Directory of the model.
            platform: Target platform to deploy the model.
            options: Additional options when deploying the model.
                Each target platform will have their own specifications of options.

        Returns:
            The deployment information.
        """
        info = Deployment(name, model_dir_path, platform, options)
        self._storage[name] = info
        return info

    def list(self) -> List[Deployment]:
        """List all deployments.

        Returns:
            A list of stored deployments information.
        """
        return list(self._storage.values())

    def get(self, name: str) -> Optional[Deployment]:
        """Get a specific deployment with the given name if exists.

        Args:
            name: Name of deployment.

        Returns:
            The deployment information. Return None if the requested deployment does not exist.
        """
        if name in self._storage:
            return self._storage[name]
        else:
            return None

    def delete(self, name: str) -> None:
        """Delete a deployment with the given name.

        Args:
            name: Name of deployment.
        """
        self._storage.pop(name)


class Deployer:
    """A deployer that deploy a model to the remote. Currently only deploying to the warehouse is supported.

    TODO(SNOW-786577): Better data modeling for deployment interface."""

    def __init__(self, session: Session, manager: DeploymentManager) -> None:
        """Initializer of the Deployer.

        Args:
            session: The session used to connect to Snowflake.
            manager: The manager used to store the deployment information.
        """
        self._manager = manager
        self._session = session

    def create_deployment(
        self,
        name: str,
        model_dir_path: str,
        platform: TargetPlatform,
        options: Dict[str, Any],
    ) -> Deployment:
        """Create a deployment and deploy it to remote platform.

        Args:
            name: Name of the deployment for the model.
            model_dir_path: Directory of the model.
            platform: Target platform to deploy the model.
            options: Additional options when deploying the model.
                Each target platform will have their own specifications of options.

        Raises:
            RuntimeError: Raised when running into issues when deploying.

        Returns:
            The deployment information.
        """
        is_success = False
        error_msg = ""
        try:
            info = self._manager.create(name, model_dir_path, platform, options)
            if platform == TargetPlatform.WAREHOUSE:
                _udf_util._deploy_to_warehouse(
                    self._session,
                    udf_name=name,
                    model_dir_path=model_dir_path,
                    relax_version=options.get("relax_version", False),
                )
            is_success = True
        except Exception as e:
            print(e)
            error_msg = str(e)
        finally:
            if not is_success:
                self._manager.delete(name)
                raise RuntimeError(error_msg)
        return info

    def list_deployments(self) -> List[Deployment]:
        """List all deployments in related deployment manager.

        Returns:
            A list of stored deployments information.
        """
        return self._manager.list()

    def get_deployment(self, name: str) -> Optional[Deployment]:
        """Get a specific deployment with the given name if exists in the related deployment manager.

        Args:
            name: Name of deployment.

        Returns:
            The deployment information. Return None if the requested deployment does not exist.
        """
        return self._manager.get(name)

    def delete_deployment(self, name: str) -> None:
        """Delete a deployment with the given name in the related deployment manager.

        Args:
            name: Name of deployment.
        """
        self._manager.delete(name)

    def predict(self, name: str, df: Union[DataFrame, pd.DataFrame]) -> pd.DataFrame:
        """Execute batch inference of a model remotely.

        Args:
            name: The name of the deployment that contains the model used to infer.
            df: The input dataframe.

        Raises:
            ValueError: Raised when the deployment does not exist.

        Returns:
            The output dataframe.
        """
        if isinstance(df, pd.DataFrame):
            df = self._session.create_dataframe(df)
        d = self.get_deployment(name)
        if not d:
            raise ValueError(f"Deployment {name} does not exist.")
        _, meta = model.load_model(d.model_dir_path)

        cols = []
        for col_name in df.columns:
            if col_name[0] == '"' and col_name[-1] == '"':
                # To deal with ugly double quoted col names
                literal_col_name = col_name[1:-1]
            else:
                literal_col_name = col_name
            cols.extend(
                [
                    type_utils.ColumnOrName(F.lit(type_utils.LiteralType(literal_col_name))),
                    type_utils.ColumnOrName(F.col(col_name)),
                ]
            )
        output_col_names = [cs.name for cs in meta.schema.outputs]
        output_cols = []
        for output_col_name in output_col_names:
            output_cols.append(F.col("tmp_result")[output_col_name].alias(output_col_name))
        return (
            df.select(F.call_udf(name, type_utils.ColumnOrLiteral(F.object_construct(*cols))).alias("tmp_result"))
            .select(*output_cols)
            .to_pandas()
        )

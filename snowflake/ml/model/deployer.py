import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import _udf_util, model_meta, model_signature, model_types
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
        model: The model object that get deployed.
        model_meta: The model metadata.
        platform: Target platform to deploy the model.
        options: Additional options when deploying the model.
    """

    name: str
    platform: TargetPlatform
    model: model_types.ModelType
    model_meta: model_meta.ModelMetadata
    options: Dict[str, str]


class DeploymentManager(ABC):
    """WIP: Intended to provide model deployment management.
    Abstract class for a deployment manager.
    """

    @abstractmethod
    def create(
        self,
        name: str,
        platform: TargetPlatform,
        model: model_types.ModelType,
        model_meta: model_meta.ModelMetadata,
        options: Dict[str, str],
    ) -> Deployment:
        """Create a deployment.

        Args:
            name: Name of the deployment for the model.
            model: The model object that get deployed.
            model_meta: The model metadata.
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

    def create(
        self,
        name: str,
        platform: TargetPlatform,
        model: model_types.ModelType,
        model_meta: model_meta.ModelMetadata,
        options: Dict[str, Any],
    ) -> Deployment:
        """Create a deployment.

        Args:
            name: Name of the deployment for the model.
            platform: Target platform to deploy the model.
            model: The model object that get deployed.
            model_meta: The model metadata.
            options: Additional options when deploying the model.
                Each target platform will have their own specifications of options.

        Returns:
            The deployment information.
        """
        info = Deployment(name, platform, model, model_meta, options)
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
    ) -> Optional[Deployment]:
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
        info = None
        try:
            if platform == TargetPlatform.WAREHOUSE:
                m, meta = _udf_util._deploy_to_warehouse(
                    self._session,
                    udf_name=name,
                    model_dir_path=model_dir_path,
                    relax_version=options.get("relax_version", False),
                )
            info = self._manager.create(name, platform, m, meta, options)
            is_success = True
        except Exception as e:
            print(e)
            error_msg = str(e)
        finally:
            if not is_success:
                if self._manager.get(name) is not None:
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

    def predict(self, name: str, df: Any) -> pd.DataFrame:
        """Execute batch inference of a model remotely.

        Args:
            name: The name of the deployment that contains the model used to infer.
            df: The input dataframe.

        Raises:
            ValueError: Raised when the deployment does not exist.
            NotImplementedError: Raised when confronting unsupported feature group.

        Returns:
            The output dataframe.
        """
        d = self.get_deployment(name)
        if not d:
            raise ValueError(f"Deployment {name} does not exist.")
        meta = d.model_meta
        if not isinstance(df, DataFrame):
            df = model_signature._validate_data_with_features_and_convert_to_df(meta.signature.inputs, df)
            s_df = self._session.create_dataframe(df)
        else:
            s_df = df

        cols = []
        for col_name in s_df.columns:
            literal_col_name = identifier.remove_quote_if_quoted(col_name)
            cols.extend(
                [
                    type_utils.ColumnOrName(F.lit(type_utils.LiteralType(literal_col_name))),
                    type_utils.ColumnOrName(F.col(col_name)),
                ]
            )
        output_col_names = [feature.name for feature in meta.signature.outputs]
        output_cols = []
        for output_col_name in output_col_names:
            # To avoid automatic upper-case convert, we quoted the result name.
            output_cols.append(F.col("tmp_result")[output_col_name].alias(f'"{output_col_name}"'))

        dtype_map = {}
        for feature in meta.signature.outputs:
            if isinstance(feature, model_signature.FeatureSpec):
                dtype_map[feature.name] = feature._dtype._value
            else:
                raise NotImplementedError("FeatureGroup is not supported yet.")
        df_local = (
            s_df.select(F.call_udf(name, type_utils.ColumnOrLiteral(F.object_construct(*cols))).alias("tmp_result"))
            .select(*output_cols)
            .to_pandas()
            .applymap(json.loads)
            .rename(columns=identifier.remove_quote_if_quoted)
            .astype(dtype=dtype_map)
        )
        return df_local

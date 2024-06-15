from typing import Any

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.dataset import dataset

_PROJECT = "Dataset"


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def create_from_dataframe(
    session: snowpark.Session,
    name: str,
    version: str,
    input_dataframe: snowpark.DataFrame,
    **version_kwargs: Any,
) -> dataset.Dataset:
    """
    Create a new versioned Dataset from a DataFrame.

    Args:
        session: The Snowpark Session instance to use.
        name: The dataset name
        version: The dataset version name
        input_dataframe: DataFrame containing data to be saved to the created Dataset.
        version_kwargs: Keyword arguments passed to dataset version creation.
            See `Dataset.create_version()` documentation for supported arguments.

    Returns:
        A Dataset object.
    """
    ds: dataset.Dataset = dataset.Dataset.create(session, name, exist_ok=True)
    ds.create_version(version, input_dataframe=input_dataframe, **version_kwargs)
    ds = ds.select_version(version)  # select_version returns a new copy
    return ds


@telemetry.send_api_usage_telemetry(project=_PROJECT)
def load_dataset(session: snowpark.Session, name: str, version: str) -> dataset.Dataset:
    """
    Load a versioned Dataset.

    Args:
        session: The Snowpark Session instance to use.
        name: The dataset name.
        version: The dataset version name.

    Returns:
        A Dataset object.
    """
    ds: dataset.Dataset = dataset.Dataset.load(session, name).select_version(version)
    return ds

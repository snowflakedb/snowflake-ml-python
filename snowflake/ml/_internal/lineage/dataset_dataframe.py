import copy
from typing import List

from snowflake import snowpark
from snowflake.ml._internal.lineage import data_source


class DatasetDataFrame(snowpark.DataFrame):
    """
    Represents a lazily-evaluated dataset. It extends :class:`snowpark.DataFrame` so all
    :class:`snowpark.DataFrame` operations can be applied to it. It holds additional information
    related to the :class`Dataset`.

    It will be created by dataset.read.to_snowpark_dataframe() API and by the transformations
    that produce a new dataframe.
    """

    @staticmethod
    def from_dataframe(
        df: snowpark.DataFrame, data_sources: List[data_source.DataSource], inplace: bool = False
    ) -> "DatasetDataFrame":
        """
        Create a new DatasetDataFrame instance from a snowpark.DataFrame instance with
        additional source information.

        Args:
            df (snowpark.DataFrame): The Snowpark DataFrame to be converted.
            data_sources (List[DataSource]): A list of data sources to associate with the DataFrame.
            inplace (bool): If True, modifies the DataFrame in place; otherwise, returns a new DatasetDataFrame.

        Returns:
            DatasetDataFrame: A new or modified DatasetDataFrame depending on the 'inplace' argument.
        """
        if not inplace:
            df = copy.deepcopy(df)
        df.__class__ = DatasetDataFrame
        df._data_sources = data_sources  # type:ignore[attr-defined]
        return df  # type: ignore[return-value]

    def _get_sources(self) -> List[data_source.DataSource]:
        """
        Returns the data sources associated with the DataFrame.
        """
        return self._data_sources  # type: ignore[no-any-return]

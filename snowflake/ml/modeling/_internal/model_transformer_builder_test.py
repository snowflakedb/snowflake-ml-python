import inflection
import pytest
from absl.testing import absltest
from sklearn.datasets import load_iris

from snowflake import snowpark
from snowflake.ml.modeling._internal.local_implementations.pandas_handlers import (
    PandasTransformHandlers,
)
from snowflake.ml.modeling._internal.model_transformer_builder import (
    ModelTransformerBuilder,
)
from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_handlers import (
    SnowparkTransformHandlers,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions


class ModelTransformBuilderTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = snowpark.Session.builder.configs(SnowflakeLoginOptions()).create()
        self._pandas_dataset = load_iris(as_frame=True).frame
        self._snowpark_dataset = self._get_snowpark_dataset()

    def tearDown(self) -> None:
        self._session.close()

    def _get_snowpark_dataset(self) -> snowpark.DataFrame:
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df: snowpark.DataFrame = self._session.create_dataframe(input_df_pandas)
        return input_df

    def test_builder_with_pd_dataset(self) -> None:
        transformer_handler = ModelTransformerBuilder.build(
            class_name="class_name", subproject="sub_project", dataset=self._pandas_dataset, estimator=None
        )
        assert isinstance(transformer_handler, PandasTransformHandlers)

    def test_builder_with_snowpark_dataset(self) -> None:
        transformer_handler = ModelTransformerBuilder.build(
            class_name="class_name", subproject="sub_project", dataset=self._snowpark_dataset, estimator=None
        )
        assert isinstance(transformer_handler, SnowparkTransformHandlers)

    def test_builder_with_invalid_dataset(self) -> None:
        dataset_json = self._pandas_dataset.to_json()
        with pytest.raises(TypeError):
            ModelTransformerBuilder.build(
                class_name="class_name", subproject="sub_project", dataset=dataset_json, estimator=None
            )


if __name__ == "__main__":
    absltest.main()

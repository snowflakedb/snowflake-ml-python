from unittest import mock

import inflection
from absl.testing import absltest
from sklearn.datasets import load_iris

from snowflake.ml.modeling.model_selection import GridSearchCV, RandomizedSearchCV
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class SearchSingleNodeTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_single_node_grid(self, mock_is_single_node) -> None:
        mock_is_single_node.return_value = True
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df = self._session.create_dataframe(input_df_pandas)

        parameters = {
            "learning_rate": [0.1],  # reduce the parameters into one to accelerate the test process
        }

        estimator = XGBClassifier()
        reg = GridSearchCV(estimator=estimator, param_grid=parameters, cv=2, verbose=True)
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)
        reg.fit(input_df)

        self.assertEqual(reg.to_sklearn(), reg._sklearn_object)

        self.assertEqual(reg._sklearn_object.n_jobs, -1)

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_single_node_random(self, mock_is_single_node) -> None:
        mock_is_single_node.return_value = True
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df = self._session.create_dataframe(input_df_pandas)

        parameters = {
            "learning_rate": [0.1],  # reduce the parameters into one to accelerate the test process
        }

        estimator = XGBClassifier()
        reg = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, cv=2, verbose=True)
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)
        reg.fit(input_df)

        self.assertEqual(reg.to_sklearn(), reg._sklearn_object)

        self.assertEqual(reg._sklearn_object.n_jobs, -1)

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_not_single_node_grid(self, mock_is_single_node) -> None:
        mock_is_single_node.return_value = False
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df = self._session.create_dataframe(input_df_pandas)

        parameters = {
            "learning_rate": [0.1],
        }

        estimator = XGBClassifier()
        reg = GridSearchCV(estimator=estimator, param_grid=parameters, cv=2, verbose=True)
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)
        reg.fit(input_df)

        self.assertEqual(reg._sklearn_object.estimator.n_jobs, 3)

    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_not_single_node_random(self, mock_is_single_node) -> None:
        mock_is_single_node.return_value = False
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        input_df_pandas["INDEX"] = input_df_pandas.reset_index().index
        input_df = self._session.create_dataframe(input_df_pandas)

        parameters = {
            "learning_rate": [0.1],  # reduce the parameters into one to accelerate the test process
        }

        estimator = XGBClassifier()
        reg = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, cv=2, verbose=True)
        reg.set_input_cols(input_cols)
        output_cols = ["OUTPUT_" + c for c in label_col]
        reg.set_output_cols(output_cols)
        reg.set_label_cols(label_col)
        reg.fit(input_df)

        self.assertEqual(reg._sklearn_object.estimator.n_jobs, 3)


if __name__ == "__main__":
    absltest.main()

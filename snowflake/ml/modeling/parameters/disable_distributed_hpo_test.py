from unittest import mock

from absl.testing import absltest
from sklearn.model_selection import GridSearchCV
from snowflake.ml.modeling.xgboost.xgb_classifier import XGBClassifier

from snowflake.ml.modeling._internal.distributed_hpo_trainer import (
    DistributedHPOTrainer,
)
from snowflake.ml.modeling._internal.model_trainer_builder import ModelTrainerBuilder
from snowflake.ml.modeling._internal.snowpark_trainer import SnowparkModelTrainer
from snowflake.snowpark import DataFrame, Session


class DisableDistributedHPOTest(absltest.TestCase):
    @mock.patch("snowflake.ml.modeling._internal.model_trainer_builder.is_single_node")
    def test_disable_distributed_hpo(self, is_single_node_mock: mock.Mock) -> None:
        is_single_node_mock.return_value = False

        mock_session = mock.MagicMock(spec=Session)
        mock_dataframe = mock.MagicMock(spec=DataFrame)
        mock_dataframe._session = mock_session

        estimator = GridSearchCV(param_grid={"max_leaf_nodes": [10, 100]}, estimator=XGBClassifier())

        trainer = ModelTrainerBuilder.build(estimator=estimator, dataset=mock_dataframe, input_cols=[])

        self.assertTrue(isinstance(trainer, DistributedHPOTrainer))

        # Disable distributed HPO
        import snowflake.ml.modeling.parameters.disable_distributed_hpo  # noqa: F401

        self.assertFalse(ModelTrainerBuilder._ENABLE_DISTRIBUTED)
        trainer = ModelTrainerBuilder.build(estimator=estimator, dataset=mock_dataframe, input_cols=[])

        self.assertTrue(isinstance(trainer, SnowparkModelTrainer))
        self.assertFalse(isinstance(trainer, DistributedHPOTrainer))


if __name__ == "__main__":
    absltest.main()

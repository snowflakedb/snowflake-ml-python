from itertools import groupby
from typing import Any

import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from absl.testing import absltest
from sklearn import datasets

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._packager.model_task import model_task_utils

binary_dataset = datasets.load_breast_cancer()
binary_data_X = pd.DataFrame(binary_dataset.data, columns=binary_dataset.feature_names)
binary_data_y = pd.Series(binary_dataset.target)
single_class_y = pd.Series([0] * len(binary_dataset.target))

multiclass_data = datasets.load_iris()
multiclass_data_X = pd.DataFrame(multiclass_data.data, columns=multiclass_data.feature_names)
multiclass_data_y = pd.Series(multiclass_data.target)

# Make a synthetic ranking dataset for demonstration
seed = 1994
ranking_X, ranking_y = datasets.make_classification(random_state=seed)
rng = np.random.default_rng(seed)
n_query_groups = 3
ranking_qid = rng.integers(0, n_query_groups, size=ranking_X.shape[0])

# Sort the inputs based on query index
sorted_idx = np.argsort(ranking_qid)
ranking_X = ranking_X[sorted_idx, :]
ranking_y = ranking_y[sorted_idx]
ranking_qid = ranking_qid[sorted_idx]


class ModelTaskUtilsTest(absltest.TestCase):
    def _validate_model_task_and_output(
        self,
        model: Any,
        expected_task: type_hints.Task,
        expected_output: model_signature.DataType,
    ) -> None:
        model_task_and_output = model_task_utils.resolve_model_task_and_output_type(model, type_hints.Task.UNKNOWN)
        self.assertEqual(expected_task, model_task_and_output.task)
        self.assertEqual(expected_output, model_task_and_output.output_type)

    def test_model_task_and_output_xgb_binary_classifier(self) -> None:
        classifier = xgboost.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        classifier.fit(binary_data_X, binary_data_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_xgb_for_single_class(self) -> None:
        # without objective
        classifier = xgboost.XGBClassifier()
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )
        # with binary objective
        classifier = xgboost.XGBClassifier(objective="binary:logistic")
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )
        # with multiclass objective
        params = {"objective": "multi:softmax", "num_class": 3}
        classifier = xgboost.XGBClassifier(**params)
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_MULTI_CLASSIFICATION, model_signature.DataType.STRING
        )

    def test_model_task_and_output_xgb_multiclass_classifier(self) -> None:
        classifier = xgboost.XGBClassifier()
        classifier.fit(multiclass_data_X, multiclass_data_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_MULTI_CLASSIFICATION, model_signature.DataType.STRING
        )

    def test_model_task_and_output_xgb_regressor(self) -> None:
        regressor = xgboost.XGBRegressor()
        regressor.fit(multiclass_data_X, multiclass_data_y)
        self._validate_model_task_and_output(
            regressor, type_hints.Task.TABULAR_REGRESSION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_xgb_booster(self) -> None:
        params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
        booster = xgboost.train(params, xgboost.DMatrix(data=binary_data_X, label=binary_data_y))
        self._validate_model_task_and_output(
            booster, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_xgb_ranker(self) -> None:
        ranker = xgboost.XGBRanker(
            tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg", lambdarank_pair_method="topk"
        )
        ranker.fit(ranking_X, ranking_y, qid=ranking_qid)
        self._validate_model_task_and_output(ranker, type_hints.Task.TABULAR_RANKING, model_signature.DataType.DOUBLE)

    def test_model_task_and_output_lightgbm_classifier(self) -> None:
        classifier = lightgbm.LGBMClassifier()
        classifier.fit(binary_data_X, binary_data_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_lightgbm_for_single_class(self) -> None:
        # without objective
        classifier = lightgbm.LGBMClassifier()
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )
        # with binary objective
        classifier = lightgbm.LGBMClassifier(objective="binary")
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )
        # with multiclass objective
        classifier = lightgbm.LGBMClassifier(objective="multiclass", num_classes=3)
        classifier.fit(binary_data_X, single_class_y)
        self._validate_model_task_and_output(
            classifier, type_hints.Task.TABULAR_MULTI_CLASSIFICATION, model_signature.DataType.STRING
        )

    def test_model_task_and_output_lightgbm_booster(self) -> None:
        booster = lightgbm.train({"objective": "binary"}, lightgbm.Dataset(binary_data_X, label=binary_data_y))
        self._validate_model_task_and_output(
            booster, type_hints.Task.TABULAR_BINARY_CLASSIFICATION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_lightgbm_regressor(self) -> None:
        regressor = lightgbm.LGBMRegressor()
        regressor.fit(multiclass_data_X, multiclass_data_y)
        self._validate_model_task_and_output(
            regressor, type_hints.Task.TABULAR_REGRESSION, model_signature.DataType.DOUBLE
        )

    def test_model_task_and_output_lightgbm_ranker(self) -> None:
        ranker = lightgbm.LGBMRanker()
        ranker.fit(ranking_X, ranking_y, group=[len(list(group)) for _, group in groupby(ranking_qid)])
        self._validate_model_task_and_output(ranker, type_hints.Task.TABULAR_RANKING, model_signature.DataType.DOUBLE)

    def test_model_task_catboost_binary_classifier(self) -> None:
        classifier = catboost.CatBoostClassifier()
        classifier.fit(binary_data_X, binary_data_y)
        self._validate_model_task_and_output(
            classifier,
            type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
            model_signature.DataType.DOUBLE,
        )

    def test_model_task_catboost_multi_classifier(self) -> None:
        classifier = catboost.CatBoostClassifier()
        classifier.fit(multiclass_data_X, multiclass_data_y)
        self._validate_model_task_and_output(
            classifier,
            type_hints.Task.TABULAR_MULTI_CLASSIFICATION,
            model_signature.DataType.STRING,
        )

    def test_model_task_catboost_ranking(self) -> None:
        ranker = catboost.CatBoostRanker()
        ranker.fit(ranking_X, ranking_y, group_id=ranking_qid)
        self._validate_model_task_and_output(
            ranker,
            type_hints.Task.TABULAR_RANKING,
            model_signature.DataType.DOUBLE,
        )

    def test_model_task_catboost_regressor(self) -> None:
        regressor = catboost.CatBoostRegressor()
        regressor.fit(multiclass_data_X, multiclass_data_y)
        self._validate_model_task_and_output(
            regressor,
            type_hints.Task.TABULAR_REGRESSION,
            model_signature.DataType.DOUBLE,
        )

    def test_model_task_and_output_unknown_model(self) -> None:
        def unknown_model(x: int) -> int:
            return x + 1

        with self.assertRaises(ValueError) as e:
            model_task_utils.resolve_model_task_and_output_type(unknown_model, type_hints.Task.UNKNOWN)
        self.assertEqual(str(e.exception), "Model type <class 'function'> is not supported")


if __name__ == "__main__":
    absltest.main()

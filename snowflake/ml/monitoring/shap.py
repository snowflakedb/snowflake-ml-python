import logging
from typing import Any, Iterable

import numpy as np

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.snowpark import functions, types

_PROJECT = "MLOps"
_SUBPROJECT = "Monitor"

logger = logging.getLogger(__name__)


class ShapExplainer:
    r"""Distributed Wrapper of Lundberg's Shap package.

    See shap's github page:
    (https://github.com/shap/shap).

    Ex:
        from snowflake.ml.modeling.metrics.monitor import ShapExplainer
        from sklearn.ensemble import RandomForestClassifier
        X_train = np.random.randint(1, 90, (4, 5))
        y_train = np.random.randint(0, 3, (4, 1))

        clf = RandomForestClassifier(max_depth=3, random_state=0)
        clf.fit(X_train, y_train)

        test_sample = np.array([[3,2,1,4,5]])

        shap_explainer = ShapExplainer(clf.predict, X_train)
        shap_values = shap_explainer1(test_sample)

    """

    def __init__(self, session: snowpark.Session, clf: Any, sample_training_data: Iterable[Any]) -> None:
        """Constructor of ShapExplainer.
        It internal constructs shap.Explainer.

        Args:
            session: the snowpark Session being used
            clf: The in mem representation of your trained model. For ex:
                clf = RandomForestClassifier(max_depth=3, random_state=0)
                clf.fit(X_train, y_train)
                some Models require you pass in clf.predict
            sample_training_data: the list or numpyArray you usually input to model.fit(X, y), the X part;
                it doesn't have to be the exact same X, but must have the same column Count;
                if it's to big, shap algo underlying will down sample.
        """
        try:
            import shap
        except ImportError:
            logger.error("To use this API, please install `shap` package in your environment.")
            return

        session.add_packages("numpy", "shap", "pandas")

        def get_shap(input: list) -> list:  # type: ignore[type-arg]
            shap_explainer = shap.Explainer(clf, sample_training_data)
            shap_values = shap_explainer(np.array([input]))
            return shap_values.values.tolist()  # type: ignore[no-any-return]

        self._shap_udf = session.udf.register(get_shap, input_types=[types.ArrayType()], return_type=types.ArrayType())

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.10")  # TODO: update versions when release
    def get_shap(self, input_df: snowpark.DataFrame) -> Any:
        """Will invoke server udf to compute shap.

        Args:
            input_df: A snowpark DataFrame representing the input. This API internally "array_construct" all columns

        Returns:
            The result Dataframe

        """
        return input_df.select(functions.array_construct(*(input_df.columns)).alias("INPUT")).select(
            self._shap_udf("INPUT").alias("SHAP")
        )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.10")  # TODO: update versions when release
    def __call__(self, input_df: snowpark.DataFrame) -> Any:
        """Will invoke server udf to compute shap.

        Args:
            input_df: A snowpark DataFrame representing the input. This API internally "array_construct" all columns

        Returns:
            The result Dataframe

        """
        return self.get_shap(input_df)

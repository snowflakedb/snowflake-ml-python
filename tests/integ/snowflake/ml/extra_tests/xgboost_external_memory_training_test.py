import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.metrics import accuracy_score as sk_accuracy_score
from xgboost import XGBClassifier as NativeXGBClassifier

from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, functions as F

categorical_columns = [
    "AGE",
    "CAMPAIGN",
    "CONTACT",
    "DAY_OF_WEEK",
    "EDUCATION",
    "HOUSING",
    "JOB",
    "LOAN",
    "MARITAL",
    "MONTH",
    "POUTCOME",
    "DEFAULT",
]
numerical_columns = [
    "CONS_CONF_IDX",
    "CONS_PRICE_IDX",
    "DURATION",
    "EMP_VAR_RATE",
    "EURIBOR3M",
    "NR_EMPLOYED",
    "PDAYS",
    "PREVIOUS",
]
label_column = ["LABEL"]
feature_cols = categorical_columns + numerical_columns + ["ROW_INDEX"]


class XGBoostExternalMemoryTrainingTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_fit_and_compare_results(self) -> None:
        input_df = (
            self._session.sql(
                """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
                FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS"""
            )
            .drop("Y")
            .withColumn("ROW_INDEX", F.monotonically_increasing_id())
        )
        pd_df = input_df.to_pandas().sort_values(by=["ROW_INDEX"])[numerical_columns + ["ROW_INDEX", "LABEL"]]
        sp_df = self._session.create_dataframe(pd_df)

        sk_reg = NativeXGBClassifier(random_state=0)
        sk_reg.fit(pd_df[numerical_columns], pd_df["LABEL"])
        sk_result = sk_reg.predict(pd_df[numerical_columns])

        sk_accuracy = sk_accuracy_score(pd_df["LABEL"], sk_result)

        reg = XGBClassifier(
            random_state=0,
            input_cols=numerical_columns,
            label_cols=label_column,
            use_external_memory_version=True,
            batch_size=10000,
        )
        reg.fit(sp_df)
        result = reg.predict(sp_df)

        result_pd = result.to_pandas().sort_values(by="ROW_INDEX")[["LABEL", "OUTPUT_LABEL"]]
        accuracy = sk_accuracy_score(result_pd["LABEL"], result_pd["OUTPUT_LABEL"])

        np.testing.assert_allclose(sk_accuracy, accuracy, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    main()

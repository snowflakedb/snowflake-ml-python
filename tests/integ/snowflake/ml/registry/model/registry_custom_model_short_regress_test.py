import decimal

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class TestRegistryCustomModelShortRegress(registry_model_test_base.RegistryModelTestBase):
    def test_custom_demo_model_decimal(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[decimal.Decimal(1.2), 2.3, 3.4], [decimal.Decimal(4.6), 2.7, 5.5]]
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1.2, 2.3, 3.4, 1.2], [4.6, 2.7, 5.5, 4.6]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_str(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}, dtype=pd.StringDtype()),
                    ),
                )
            },
        )


if __name__ == "__main__":
    absltest.main()

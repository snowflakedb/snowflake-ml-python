import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias"), encoding="utf-8") as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class MultipleModelTest(registry_model_test_base.RegistryModelTestBase):
    def test_multiple_model(self) -> None:
        version = "v1"
        arr = np.array([[1], [4]])
        pd_df = pd.DataFrame(arr, columns=["c1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm_1 = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            name_1 = f"model_{self._run_id}_1"

            mv1 = self.registry.log_model(lm_1, model_name=name_1, version_name=version, sample_input_data=pd_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("20")
            lm_2 = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            name_2 = f"model_{self._run_id}_2"

            mv2 = self.registry.log_model(lm_2, model_name=name_2, version_name=version, sample_input_data=pd_df)

        res = (
            self.session.sql(f"SELECT {name_1}!predict(1):output as A, {name_2}!predict(1):output as B")
            .collect()[0]
            .as_dict()
        )

        self.assertDictEqual(res, {"A": "11", "B": "21"})

        res = (
            mv1.run(mv2.run(self.session.create_dataframe(pd_df)).select('"output"').rename({'"output"': '"c1"'}))
            .select('"output"')
            .to_pandas()
        )
        pd.testing.assert_frame_equal(res, pd.DataFrame({"output": [31, 34]}))


if __name__ == "__main__":
    absltest.main()

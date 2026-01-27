import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import JobSpec, OutputSpec
from snowflake.ml.model._signatures import numpy_handler
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


def _prepare_keras_functional_model() -> tuple[keras.Model, npt.ArrayLike, npt.ArrayLike]:
    n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
    x = np.random.rand(batch_size, n_input)
    y = np.random.random_integers(0, 1, (batch_size,)).astype(np.float32)

    input = keras.Input(shape=(n_input,))
    input_2 = keras.layers.Dense(n_hidden, activation="relu")(input)
    output = keras.layers.Dense(n_out, activation="sigmoid")(input_2)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError())
    model.fit(x, y, batch_size=batch_size, epochs=100)
    return model, x, y


class TestKerasBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @parameterized.parameters(  # type: ignore[misc]
        {"gpu_requests": None, "cpu_requests": None, "memory_requests": None},
        # uncomment this after https://snowflakecomputing.atlassian.net/browse/SNOW-2369772 is fixed
        # {"gpu_requests": "1", "cpu_requests": None, "memory_requests": "4Gi"},
    )
    def test_keras(
        self,
        gpu_requests: str,
        cpu_requests: str,
        memory_requests: str,
    ) -> None:
        model, data_x, data_y = _prepare_keras_functional_model()
        x_df = numpy_handler.NumpyArrayHandler.convert_to_df(data_x)
        x_df.columns = [f"input_feature_{i}" for i in range(len(x_df.columns))]

        # Generate expected predictions using the original model
        model_output = model.predict(data_x)
        model_output_df = pd.DataFrame({"output_feature_0": model_output.flatten()})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(x_df, model_output_df)

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=x_df,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                gpu_requests=gpu_requests,
                cpu_requests=cpu_requests,
                memory_requests=memory_requests,
                num_workers=1,
                replicas=2,
                function_name="predict",
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()

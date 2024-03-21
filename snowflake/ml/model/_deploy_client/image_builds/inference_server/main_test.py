import contextlib
import http
import os

import pandas as pd
import sklearn.datasets as datasets
import sklearn.neighbors as neighbors
from absl.testing import absltest
from absl.testing.absltest import mock
from starlette import testclient

from snowflake.ml._internal import file_utils
from snowflake.ml.model import custom_model
from snowflake.ml.model._packager import model_packager


class MainTest(absltest.TestCase):
    """
    This test utilizes TestClient, powered by httpx, to send requests to the Starlette application.
    """

    def setUp(self) -> None:
        super().setUp()
        self.model_zip_path = self.setup_model()

    def setup_model(self) -> str:
        iris = datasets.load_iris(as_frame=True)
        x = iris.data
        y = iris.target
        knn_model = neighbors.KNeighborsClassifier()
        knn_model.fit(x, y)

        class TestCustomModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(knn_model.predict(input))

        model = TestCustomModel(custom_model.ModelContext())
        tmpdir = self.create_tempdir()
        tmpdir_for_zip = self.create_tempdir()
        zip_full_path = os.path.join(tmpdir_for_zip.full_path, "model.zip")
        model_packager.ModelPackager(tmpdir.full_path).save(
            name="test_model",
            model=model,
            sample_input_data=x,
            metadata={"author": "halu", "version": "1"},
        )
        file_utils.make_archive(zip_full_path, tmpdir.full_path)
        return zip_full_path

    def test_setup_import(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TARGET_METHOD": "predict",
                "MODEL_ZIP_STAGE_PATH": self.model_zip_path,
            },
        ):
            with mock.patch.object(
                model_packager.ModelPackager,
                "load",
                side_effect=ImportError("Cannot import transformers", name="transformers"),
            ):
                from main import _run_setup

                with self.assertRaisesRegex(ImportError, "Cannot import transformers"):
                    _run_setup()

    @contextlib.contextmanager
    def common_helper(self):  # type: ignore[no-untyped-def]
        with mock.patch.dict(
            os.environ,
            {
                "TARGET_METHOD": "predict",
                "MODEL_ZIP_STAGE_PATH": self.model_zip_path,
            },
        ):
            import main

            client = testclient.TestClient(main.app)
            yield main, client

    def test_ready_endpoint_after_model_successfully_loaded(self) -> None:
        with self.common_helper() as (_, client):
            response = client.get("/health")
            self.assertEqual(response.status_code, http.HTTPStatus.OK)
            self.assertEqual(response.json(), {"status": "ready"})

    def test_ready_endpoint_during_model_loading(self) -> None:
        with self.common_helper() as (main, client):
            with mock.patch("main._MODEL_LOADING_STATE", main._ModelLoadingState.LOADING):
                response = client.get("/health")
                self.assertEqual(response.status_code, http.HTTPStatus.SERVICE_UNAVAILABLE)
                self.assertEqual(response.json(), {"status": "not ready"})

    def test_ready_endpoint_after_model_loading_failed(self) -> None:
        with self.common_helper() as (main, client):
            with mock.patch("main._MODEL_LOADING_STATE", main._ModelLoadingState.FAILED):
                response = client.get("/health")
                self.assertEqual(response.status_code, http.HTTPStatus.SERVICE_UNAVAILABLE)
                self.assertEqual(response.json(), {"status": "not ready"})

    def test_predict_endpoint_happy_path(self) -> None:
        with self.common_helper() as (_, client):
            # Construct data input based on external function data input format
            data = {
                "data": [
                    [
                        0,
                        {
                            "_ID": 0,
                            "sepal length (cm)": 5.1,
                            "sepal width (cm)": 3.5,
                            "petal length (cm)": 4.2,
                            "petal width (cm)": 1.3,
                        },
                    ],
                    [
                        1,
                        {
                            "_ID": 1,
                            "sepal length (cm)": 4.7,
                            "sepal width (cm)": 3.2,
                            "petal length (cm)": 4.1,
                            "petal width (cm)": 4.2,
                        },
                    ],
                ]
            }

            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, http.HTTPStatus.OK)
            expected_response = {
                "data": [[0, {"output_feature_0": 1, "_ID": 0}], [1, {"output_feature_0": 2, "_ID": 1}]]
            }
            self.assertEqual(response.json(), expected_response)

    def test_predict_endpoint_with_invalid_input(self) -> None:
        with self.common_helper() as (_, client):
            response = client.post("/predict", json={})
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, "Input data malformed: missing data field in the request input")

            response = client.post("/predict", json={"data": []})
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, "Input data malformed")

            # Input data with indexes only.
            response = client.post("/predict", json={"data": [[0], [1]]})
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, "Input data malformed")

            response = client.post(
                "/predict",
                json={
                    "foo": [
                        [1, 2],
                        [2, 3],
                    ]
                },
            )
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, "Input data malformed: missing data field in the request input")

    def test_predict_with_misshaped_data(self) -> None:
        with self.common_helper() as (_, client):
            data = {
                "data": [
                    [
                        0,
                        {
                            "_ID": 0,
                            "sepal length (cm)": 5.1,
                            "sepal width (cm)": 3.5,
                            "petal length (cm)": 4.2,
                        },
                    ],
                    [
                        1,
                        {
                            "_ID": 1,
                            "sepal length (cm)": 4.7,
                            "sepal width (cm)": 3.2,
                            "petal length (cm)": 4.1,
                        },
                    ],
                ]
            }

            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, r"Input data malformed: .*dtype mappings argument.*")

    def test_predict_with_incorrect_data_type(self) -> None:
        with self.common_helper() as (_, client):
            data = {
                "data": [
                    [
                        0,
                        {
                            "_ID": 0,
                            "sepal length (cm)": "a",
                            "sepal width (cm)": "b",
                            "petal length (cm)": "c",
                            "petal width (cm)": "d",
                        },
                    ]
                ]
            }
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)
            self.assertRegex(response.text, "Input data malformed: could not convert string to float")


if __name__ == "__main__":
    absltest.main()

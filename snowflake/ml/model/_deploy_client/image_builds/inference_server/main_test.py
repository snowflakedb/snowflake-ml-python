import pandas as pd
import sklearn.datasets as datasets
import sklearn.neighbors as neighbors
from absl.testing import absltest
from absl.testing.absltest import mock
from starlette import testclient

from snowflake.ml.model import custom_model


class MainTest(absltest.TestCase):
    """
    This test utilizes TestClient, powered by httpx, to send requests to the Starlette application. It optionally skips
    the model loading step in the inference code, which is irrelevant for route testing and challenging to mock due to
    gunicorn's preload option when loading the Starlette Python app. This skipping is achieved by checking the presence
    of the 'PYTEST_CURRENT_TEST' environment variable during pytest execution, the 'TEST_WORKSPACE' variable during
    bazel test execution, or the 'TEST_SRCDIR' variable during Absl test execution.
    """

    def setUp(self) -> None:
        super().setUp()

        from main import app

        self.client = testclient.TestClient(app)

        self.loaded_model = self.get_custom_model()

    def get_custom_model(self) -> custom_model.CustomModel:
        # Set up a mock model
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

        return TestCustomModel(custom_model.ModelContext())

    def test_ready_endpoint(self) -> None:
        with mock.patch("main.loaded_model", return_value=self.loaded_model):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ready"})

    def test_predict_endpoint_happy_path(self) -> None:
        data = {
            "data": [[0, 5.1, 3.5, 4.2, 1.3], [1, 4.7, 3.2, 4.1, 4.2], [2, 5.1, 3.5, 4.2, 4.6], [3, 4.7, 3.2, 4.1, 5.1]]
        }

        with mock.patch("main.loaded_model", self.loaded_model):
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)
        expected_response = {"data": [[0, 1], [1, 2], [2, 2], [3, 2]]}
        self.assertEqual(response.json(), expected_response)

    def test_predict_endpoint_with_invalid_input(self) -> None:
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertRegex(response.text, "Input data malformed: missing data field in the request input")

        response = self.client.post("/predict", json={"data": []})
        self.assertEqual(response.status_code, 400)
        self.assertRegex(response.text, "Input data malformed: empty data")

        # Input data with indexes only.
        response = self.client.post("/predict", json={"data": [[0], [1]]})
        self.assertEqual(response.status_code, 400)
        self.assertRegex(response.text, "Input data malformed: empty data")

        response = self.client.post(
            "/predict",
            json={
                "foo": [
                    [1, 2],
                    [2, 3],
                ]
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertRegex(response.text, "Input data malformed: missing data field in the request input")

    def test_predict_with_misshaped_data(self) -> None:
        data = {"data": [[0, 5.1, 3.5, 4.2], [1, 4.7, 3.2, 4.1], [2, 5.1, 3.5, 4.2], [3, 4.7, 3.2, 4.1]]}

        with mock.patch("main.loaded_model", self.loaded_model):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertRegex(
                response.text,
                "Prediction failed: X has 3 features, but KNeighborsClassifier is " "expecting 4 features as input",
            )

    def test_predict_with_incorrect_data_type(self) -> None:
        data = {
            "data": [
                [0, "a", "b", "c", "d"],
            ]
        }

        with mock.patch("main.loaded_model", self.loaded_model):
            response = self.client.post("/predict", json=data)
            self.assertEqual(response.status_code, 400)
            self.assertRegex(response.text, "Prediction failed: could not convert string to float")


if __name__ == "__main__":
    absltest.main()

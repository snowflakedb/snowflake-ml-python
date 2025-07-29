from absl.testing import absltest

from snowflake.ml.experiment import utils


class ExperimentTrackingUtilsTest(absltest.TestCase):
    def test_flatten_nested_params_dict(self) -> None:
        """Test that nested dictionary parameters are flattened correctly."""
        nested_params = {"param1": 1, "param2": {"sub_param1": 2, "sub_param2": {"sub_sub_param1": 3}}}
        expected_flattened = {
            "param1": 1,
            "param2.sub_param1": 2,
            "param2.sub_param2.sub_sub_param1": 3,
        }
        flattened = utils.flatten_nested_params(nested_params)
        self.assertEqual(flattened, expected_flattened)

    def test_flatten_nested_params_list(self) -> None:
        """Test that nested list parameters are flattened correctly."""
        nested_params = [1, [2, [3]]]
        expected_flattened = {
            "0": 1,
            "1.0": 2,
            "1.1.0": 3,
        }
        flattened = utils.flatten_nested_params(nested_params)
        self.assertEqual(flattened, expected_flattened)

    def test_flatten_nested_params_mixed(self) -> None:
        """Test that mixed nested parameters (dict and list) are flattened correctly."""
        nested_params = {
            "param1": 1,
            "param2": [2, {"sub_param1": 3}, ["four", 5]],
        }
        expected_flattened = {
            "param1": 1,
            "param2.0": 2,
            "param2.1.sub_param1": 3,
            "param2.2.0": "four",
            "param2.2.1": 5,
        }
        flattened = utils.flatten_nested_params(nested_params)
        self.assertEqual(flattened, expected_flattened)

    def test_flatten_nested_params_replace_dot(self) -> None:
        """Test that dots in keys are replaced to avoid collisions."""
        nested_params = {"param.with.dot": 1, "another.param": {"nested.with.dot": 2}}
        expected_flattened = {
            "param_with_dot": 1,
            "another_param.nested_with_dot": 2,
        }
        flattened = utils.flatten_nested_params(nested_params)
        self.assertEqual(flattened, expected_flattened)


if __name__ == "__main__":
    absltest.main()

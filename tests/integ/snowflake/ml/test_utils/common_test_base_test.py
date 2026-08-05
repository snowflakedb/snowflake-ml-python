from unittest import mock

from absl.testing import absltest, parameterized

from tests.integ.snowflake.ml.test_utils import common_test_base


class CommonTestBaseTest(common_test_base.CommonTestBase):
    @common_test_base.CommonTestBase.sproc_test()
    @parameterized.parameters({"content": "hello"}, {"content": "snowflake"})  # type: ignore[misc]
    def test_parameterized_dict(self, content: str) -> None:
        print(content)

    @common_test_base.CommonTestBase.sproc_test()
    @parameterized.named_parameters(  # type: ignore[misc]
        {"testcase_name": "Normal", "content": "hello"}, {"testcase_name": "Surprise", "content": "snowflake"}
    )
    def test_named_parameterized_dict(self, content: str) -> None:
        print(content)


class HfHubDownloadRetryHelperTest(absltest.TestCase):
    def test_wrap_with_hf_hub_retry_retries_connection_error(self) -> None:
        mock_fn = mock.Mock(side_effect=[ConnectionError("connection aborted"), "ok"])
        wrapped = common_test_base._wrap_with_hf_hub_retry(mock_fn)
        with mock.patch("time.sleep") as mock_sleep:
            result = wrapped("arg", key="value")

        self.assertEqual(result, "ok")
        self.assertEqual(mock_fn.call_count, 2)
        mock_fn.assert_called_with("arg", key="value")
        mock_sleep.assert_called()

    def test_wrap_with_hf_hub_retry_does_not_retry_value_error(self) -> None:
        mock_fn = mock.Mock(side_effect=ValueError("bad input"))
        wrapped = common_test_base._wrap_with_hf_hub_retry(mock_fn)
        with mock.patch("time.sleep") as mock_sleep, self.assertRaises(ValueError):
            wrapped()

        self.assertEqual(mock_fn.call_count, 1)
        mock_sleep.assert_not_called()

    def test_wrap_with_hf_hub_retry_raises_after_max_attempts(self) -> None:
        mock_fn = mock.Mock(side_effect=ConnectionError("connection aborted"))
        wrapped = common_test_base._wrap_with_hf_hub_retry(mock_fn)
        with mock.patch("time.sleep") as mock_sleep, self.assertRaises(ConnectionError):
            wrapped()

        self.assertEqual(mock_fn.call_count, common_test_base._HF_HUB_RETRY_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, common_test_base._HF_HUB_RETRY_ATTEMPTS - 1)


if __name__ == "__main__":
    absltest.main()

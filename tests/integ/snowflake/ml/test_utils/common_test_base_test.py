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


if __name__ == "__main__":
    absltest.main()

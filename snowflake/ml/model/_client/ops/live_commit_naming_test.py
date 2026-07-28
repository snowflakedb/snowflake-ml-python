import re

from absl.testing import absltest

from snowflake.ml.model._client.ops import live_commit_naming

_PENDING_MODEL_NAME_PATTERN = re.compile(r"^PENDING_[0-9A-F]{8}_MODEL$")
_LIVE_VERSION_NAME_PATTERN = re.compile(r"^LIVE_[0-9A-F]{8}_VERSION$")


def _is_pending_model_name(name: str) -> bool:
    return _PENDING_MODEL_NAME_PATTERN.match(name) is not None


def _is_live_version_name(name: str) -> bool:
    return _LIVE_VERSION_NAME_PATTERN.match(name) is not None


class LiveCommitNamingTest(absltest.TestCase):
    def test_generate_pending_model_name_format(self) -> None:
        name = live_commit_naming.generate_pending_model_name().resolved()
        self.assertTrue(_is_pending_model_name(name))

    def test_generate_live_version_name_format(self) -> None:
        name = live_commit_naming.generate_live_version_name().resolved()
        self.assertTrue(_is_live_version_name(name))

    def test_generated_names_are_unique(self) -> None:
        pending_names = {live_commit_naming.generate_pending_model_name().resolved() for _ in range(20)}
        live_names = {live_commit_naming.generate_live_version_name().resolved() for _ in range(20)}
        self.assertLen(pending_names, 20)
        self.assertLen(live_names, 20)


if __name__ == "__main__":
    absltest.main()

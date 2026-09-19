from typing import NamedTuple

from absl.testing import absltest, parameterized

from tests.integ.snowflake.ml.jobs import job_test_base


class _RuntimeLine(NamedTuple):
    """The newest runtime the deployment registers, split into the parts a selector can pin."""

    image: str
    repo: str
    version: str
    minor: str
    python_suffix: str


class RuntimeEnvIntegrationTest(job_test_base.JobTestBase):
    """Covers how the backend resolves a runtime selector, without submitting a job.

    Job submission is covered by jobs_integ_test.py. Resolution is tested separately because the
    selector shapes worth covering include ones no job could run, such as an image tag that is not
    published.
    """

    def _newest_runtime_line(self) -> _RuntimeLine:
        """Read the newest registered runtime back from the deployment.

        Expectations are derived from this rather than hardcoded, since each deployment publishes its
        own runtime images and a new release would otherwise break these tests.
        """
        image = self._resolve_for_local_python()
        repo, _, tag = image.rpartition(":")
        # e.g. "2.9.0-py311" -> version "2.9.0", suffix "-py311". The suffix is empty on the line's
        # default Python, which keeps every assertion below correct either way.
        version = tag.split("-")[0]
        return _RuntimeLine(
            image=image,
            repo=repo,
            version=version,
            minor=".".join(version.split(".")[:2]),
            python_suffix=tag[len(version) :],
        )

    @parameterized.named_parameters(  # type: ignore[misc]
        ("bare_minor_version", "{minor}"),
        ("published_patch_version", "{version}"),
        ("older_patch_version", "{minor}.0"),
        ("unpublished_patch_version", "{minor}.99"),
        ("python_suffixed_patch_version", "{minor}.0{python_suffix}"),
    )
    def test_version_pin_resolves_to_newest_patch(self, pin_template: str) -> None:
        # Resolution keys off major.minor, so every spelling of the same line lands on the same image.
        # A pin that already carries its own Python suffix keeps that suffix while the patch still moves
        # forward, i.e. "<line>.0-py311" resolves to the newest "<line>.<patch>-py311".
        runtime_line = self._newest_runtime_line()
        pinned_version = pin_template.format(**runtime_line._asdict())
        self.assertEqual(runtime_line.image, self._resolve_for_local_python(pinned_version))

    def test_full_image_path_resolves_like_a_bare_version(self) -> None:
        runtime_line = self._newest_runtime_line()
        self.assertEqual(
            self._resolve_runtime_image(runtime_environment=runtime_line.minor),
            self._resolve_runtime_image(runtime_environment=f"{runtime_line.repo}:{runtime_line.minor}.0"),
        )

    def test_full_image_path_with_python_version_resolves_like_a_version_pin(self) -> None:
        # Same equivalence as above, but through the selector the client actually builds once a Python
        # version is in play: a full image path and the bare version it points at resolve alike whether
        # or not a Python version rides along.
        runtime_line = self._newest_runtime_line()
        full_image_path = f"{runtime_line.repo}:{runtime_line.minor}.0"
        self.assertEqual(
            self._resolve_for_local_python(runtime_line.minor),
            self._resolve_for_local_python(full_image_path),
        )


if __name__ == "__main__":
    absltest.main()

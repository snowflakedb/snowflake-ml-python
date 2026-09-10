from absl.testing import absltest

from snowflake.ml import jobs
from tests.integ.snowflake.ml.jobs.job_test_base import JobTestBase
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset


class ArcticTrainingTest(JobTestBase):
    def test_job_arctic_training(self) -> None:
        """Test that arctic-training can be run as a job."""
        rows = self.session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'ALLOW_ALL%'").collect()
        if not rows:
            self.skipTest("No compatible EAIs found in environment.")
        allow_all_eais = [r["name"] for r in rows]

        job = jobs.submit_directory(
            TestAsset("train_recipes").path,
            self.compute_pool,
            stage_name="payload_stage",
            entrypoint=["arctic_training", "run_causal.yml"],
            # Pin arctic-training and deepspeed: 0.7.1 still allows
            # deepspeed>=0.18.2, and newer DeepSpeed registers torch custom
            # ops with PEP 585 annotations (`list[int]`). The job image's
            # torch.library.infer_schema only accepts typing.List[int].
            pip_requirements=["arctic-training==0.7.1", "deepspeed==0.18.2"],
            external_access_integrations=allow_all_eais,
            session=self.session,
        )

        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))
        self.assertIn("arctic", job.get_logs(verbose=True).lower())


if __name__ == "__main__":
    absltest.main()

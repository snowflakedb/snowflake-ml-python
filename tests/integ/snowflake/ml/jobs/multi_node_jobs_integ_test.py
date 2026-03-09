import re
from typing import Optional

from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from tests.integ.snowflake.ml.jobs.job_test_base import JobTestBase
from tests.integ.snowflake.ml.jobs.test_file_helper import TestAsset


def dummy_function() -> None:
    print("hello world")


class MultiNodeJobsTest(JobTestBase):
    def test_multinode_job_basic(self) -> None:
        job_from_file = self._submit_func_as_file(dummy_function, target_instances=2)

        self.assertEqual(job_from_file.target_instances, 2)
        self.assertEqual(job_from_file.min_instances, 2)  # min_instances defaults to target_instances
        self.assertEqual(job_from_file.wait(), "DONE", file_job_logs := job_from_file.get_logs())
        self.assertIn("hello world", file_job_logs)

        @jobs.remote(self.compute_pool, stage_name="payload_stage", target_instances=2, session=self.session)
        def dummy_remote_multinode() -> None:
            print("hello world")

        job_from_func = dummy_remote_multinode()
        self.assertEqual(job_from_func.wait(), "DONE", file_job_logs := job_from_func.get_logs())
        self.assertIn("hello world", file_job_logs)

    def test_multinode_job_ray_task(self) -> None:
        def ray_workload() -> None:
            import socket

            import ray

            @ray.remote(scheduling_strategy="SPREAD")
            def compute_heavy(n: int) -> str:
                # a quick CPU‐bound toy workload
                # Create a large matrix and perform expensive operations
                import numpy as np

                matrix = np.random.rand(500, 500)
                for _ in range(n):
                    # Matrix multiplication and other expensive operations
                    matrix = np.matmul(matrix, matrix.T)
                    matrix = np.linalg.qr(matrix)[0]  # QR decomposition
                    matrix = np.sin(matrix)  # Element-wise operations
                # report which node we ran on
                return socket.gethostname()

            ray.init(address="auto", ignore_reinit_error=True)
            hosts = [compute_heavy.remote(20) for _ in range(10)]
            unique_hosts = set(ray.get(hosts))
            assert (
                len(unique_hosts) >= 2
            ), f"Expected at least 2 unique hosts, get: {unique_hosts}, hosts: {ray.get(hosts)}"
            print("test succeeded")

        job = self._submit_func_as_file(ray_workload, target_instances=2, min_instances=2)
        self.assertEqual(job.wait(), "DONE", f"job {job.id} logs: {job.get_logs(verbose=True)}")
        self.assertTrue("test succeeded" in job.get_logs())

    def test_multinode_job_wait_for_instances(self) -> None:
        def get_cluster_size() -> None:
            from common_utils import common_util as mlrs_util

            num_nodes = mlrs_util.get_num_ray_nodes()
            print("num_nodes:", num_nodes)

        # Verify min_instances met
        job1 = self._submit_func_as_file(get_cluster_size, target_instances=3, min_instances=2)
        self.assertEqual(job1.target_instances, 3)
        self.assertEqual(job1.min_instances, 2)
        self.assertEqual(job1.wait(), "DONE", job1.get_logs(verbose=True))
        self.assertIsNotNone(
            match_group := re.search(r"num_nodes: (\d+)", concise_logs := job1.get_logs(verbose=False)),
            concise_logs,
        )
        self.assertBetween(int(match_group.group(1)), 2, 3, match_group.groups())

        # Check verbose log to ensure min_instances was checked
        self.assertIn("instance requirement met", job1.get_logs(verbose=True))

        # Verify min_wait is respected
        job2 = self._submit_func_as_file(
            get_cluster_size,
            target_instances=2,
            min_instances=1,
            env_vars={"MLRS_INSTANCES_MIN_WAIT": 720},
        )
        self.assertEqual(job2.target_instances, 2)
        self.assertEqual(job2.min_instances, 1)
        while job2.status != "RUNNING":
            import time

            time.sleep(2)
        self.assertIsNotNone(job2.get_ray_dashboard_url(), job2.get_logs(verbose=True))
        self.assertEqual(job2.wait(), "DONE", job2.get_logs(verbose=True))
        self.assertIsNotNone(
            match_group := re.search(r"num_nodes: (\d+)", concise_logs := job2.get_logs(verbose=False)),
            concise_logs,
        )
        self.assertEqual(int(match_group.group(1)), 2)

    def test_min_instances_exceeding_max_nodes(self) -> None:
        compute_pool_info = self.dbm.show_compute_pools(self.compute_pool).collect()
        self.assertTrue(compute_pool_info, f"Could not find compute pool {self.compute_pool}")
        max_nodes = int(compute_pool_info[0]["max_nodes"])

        # Calculate a min_instances value that exceeds max_nodes
        min_instances = max_nodes + 1
        # Set target_instances to be greater than min_instances to pass the first validation
        target_instances = min_instances + 1

        # Attempt to submit a job with min_instances exceeding max_nodes
        with self.assertRaisesRegex(ValueError, "min_instances .* exceeds the max_nodes"):
            self._submit_func_as_file(
                dummy_function,
                min_instances=min_instances,
                target_instances=target_instances,
            )

    @parameterized.parameters(  # type: ignore[misc]
        ("src/multinode_import_zip.py", "src/test_data_processor.zip", "data_processor"),
        ("src/multinode_import_module.py", "src/subdir/utils", "src.subdir.utils"),
    )
    def test_multinode_import(self, entrypoint: str, import_path: str, import_name: Optional[str]) -> None:
        """Test that imports work on worker nodes in multi-node jobs."""
        job = jobs.submit_file(
            TestAsset(entrypoint).path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            imports=[(TestAsset(import_path).path, import_name)],
            target_instances=2,
            min_instances=2,
        )
        self.assertEqual(job.wait(), "DONE", job.get_logs(verbose=True))


if __name__ == "__main__":
    absltest.main()

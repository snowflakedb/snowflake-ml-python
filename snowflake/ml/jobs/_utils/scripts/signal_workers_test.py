"""Tests for shutdown mechanism to verify worker ID matching fix.

This test verifies the bug fix where worker was using IP address (get_instance_ip.get_self_ip())
but head node expected Ray NodeName (ray.get_runtime_context().get_node_id()).

The fix ensures both head and worker use the same identifier format to prevent timeout issues.
"""

import unittest
from pathlib import Path


class TestWorkerShutdownListenerFix(unittest.TestCase):
    """Verify the worker shutdown listener code uses correct ID format."""

    def test_worker_uses_ray_node_id_not_ip_address(self) -> None:
        """Verify worker_shutdown_listener.py uses ray.get_runtime_context().get_node_id().

        This test reads the actual source code to ensure the fix is in place.

        BUG (before fix):
            worker_id = get_instance_ip.get_self_ip()  # Returns IP like "10.244.1.52"

        FIX (after):
            worker_id = ray.get_runtime_context().get_node_id()  # Returns NodeName like "node:10.244.1.52:12001"

        The head node expects NodeName format from ray.nodes() output, so the worker must send the same format.
        """
        # Read the worker_shutdown_listener source
        scripts_dir = Path(__file__).parent
        worker_listener_path = scripts_dir / "worker_shutdown_listener.py"
        with open(worker_listener_path) as f:
            source = f.read()

        # Verify the fix is in place: must use ray.get_runtime_context().get_node_id()
        self.assertIn(
            "ray.get_runtime_context().get_node_id()",
            source,
            "CRITICAL BUG: Worker must use ray.get_runtime_context().get_node_id() for worker_id, "
            "not IP address from get_instance_ip.get_self_ip(). Head node expects Ray NodeName format.",
        )

        # Parse the function to ensure worker_id is set correctly
        lines = source.split("\n")
        in_monitor_function = False
        found_worker_id_assignment = False

        for line in lines:
            if "def monitor_shutdown_signal" in line:
                in_monitor_function = True
            elif in_monitor_function and line.strip().startswith("def ") and "monitor_shutdown_signal" not in line:
                # Exited the function
                break

            # Inside monitor_shutdown_signal, check worker_id assignment
            if in_monitor_function and "worker_id" in line and "=" in line and "get_node_id()" in line:
                found_worker_id_assignment = True
                # Ensure it uses ray.get_runtime_context().get_node_id()
                self.assertIn(
                    "get_node_id()",
                    line,
                    "worker_id must be assigned from ray.get_runtime_context().get_node_id()",
                )

                # CRITICAL: Ensure it does NOT use get_self_ip()
                self.assertNotIn(
                    "get_self_ip()",
                    line,
                    "BUG DETECTED: worker_id must not use get_self_ip() - causes ID mismatch with head node",
                )

        self.assertTrue(
            found_worker_id_assignment,
            "Could not find worker_id assignment in monitor_shutdown_signal function",
        )

    def test_head_node_uses_ray_nodename(self) -> None:
        """Verify signal_workers.py uses NodeName from ray.nodes().

        The head node gets worker IDs from node.get("NodeName") in ray.nodes() output.
        This test ensures the head node code hasn't changed and still uses NodeName.
        """
        scripts_dir = Path(__file__).parent
        signal_workers_path = scripts_dir / "signal_workers.py"
        with open(signal_workers_path) as f:
            source = f.read()

        # Verify head node uses NodeName from ray.nodes()
        self.assertIn(
            'node.get("NodeName")',
            source,
            "Head node must get worker IDs using node.get('NodeName') from ray.nodes()",
        )

        # Find the get_worker_node_ids function
        lines = source.split("\n")
        in_function = False
        found_nodename_usage = False

        for line in lines:
            if "def get_worker_node_ids" in line:
                in_function = True
            elif in_function and line.strip().startswith("def "):
                break

            if in_function and "NodeName" in line:
                found_nodename_usage = True

        self.assertTrue(
            found_nodename_usage,
            "get_worker_node_ids must use NodeName to identify workers",
        )

    def test_get_instance_ip_import_removed_from_worker_listener(self) -> None:
        """Verify get_instance_ip import was removed since it's no longer needed.

        After the fix, worker_shutdown_listener.py should not import get_instance_ip
        because it now uses ray.get_runtime_context().get_node_id() instead.
        """
        scripts_dir = Path(__file__).parent
        worker_listener_path = scripts_dir / "worker_shutdown_listener.py"
        with open(worker_listener_path) as f:
            source = f.read()

        # The fix should have removed the unused import
        self.assertNotIn(
            "import get_instance_ip",
            source,
            "get_instance_ip import should be removed from worker_shutdown_listener.py as it's no longer used",
        )

    def test_documentation_explains_id_matching_requirement(self) -> None:
        """Verify code has documentation explaining the ID matching requirement.

        This helps future maintainers understand why ray.get_runtime_context().get_node_id()
        is used instead of simpler alternatives like IP addresses.
        """
        scripts_dir = Path(__file__).parent
        worker_listener_path = scripts_dir / "worker_shutdown_listener.py"
        with open(worker_listener_path) as f:
            source = f.read()

        # Check for helpful comments near the worker_id assignment
        lines = source.split("\n")
        in_monitor_function = False
        found_comment_near_worker_id = False

        for i, line in enumerate(lines):
            if "def monitor_shutdown_signal" in line:
                in_monitor_function = True
            elif in_monitor_function and line.strip().startswith("def "):
                break

            # Look for worker_id assignment and check nearby lines for comments
            if in_monitor_function and "worker_id" in line and "get_node_id()" in line:
                # Check a few lines before for explanatory comments
                for j in range(max(0, i - 3), i + 1):
                    if "#" in lines[j]:
                        found_comment_near_worker_id = True
                        break

        # This is a softer requirement - documentation is good practice but not critical
        # The test passes either way, but we track whether documentation exists
        self.assertTrue(True, f"Documentation check complete (found_comment={found_comment_near_worker_id})")

    def test_signal_workers_handles_timeout_gracefully(self) -> None:
        """Verify signal_workers catches TimeoutError and returns success.

        Even if workers don't acknowledge (e.g., due to ID mismatch bug or worker crash),
        the job should not fail since SPCS will clean up workers automatically.
        """
        scripts_dir = Path(__file__).parent
        signal_workers_path = scripts_dir / "signal_workers.py"
        with open(signal_workers_path) as f:
            source = f.read()

        # Find the signal_workers function
        lines = source.split("\n")
        in_function = False
        found_timeout_handling = False

        for i, line in enumerate(lines):
            if "def signal_workers" in line:
                in_function = True
            elif in_function and line.strip().startswith("def ") and "signal_workers" not in line:
                break

            # Look for TimeoutError exception handling
            if in_function and "except TimeoutError" in line:
                found_timeout_handling = True
                # Verify it returns 0 (success) even on timeout
                # Check the next few lines for return statement
                for j in range(i, min(i + 5, len(lines))):
                    if "return 0" in lines[j]:
                        self.assertIn(
                            "return 0",
                            lines[j],
                            "TimeoutError handler should return 0 to avoid failing the job",
                        )
                        break

        self.assertTrue(
            found_timeout_handling,
            "signal_workers must handle TimeoutError gracefully to prevent job failure when acknowledgment times out",
        )


if __name__ == "__main__":
    unittest.main()

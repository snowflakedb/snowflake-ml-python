"""Test that zip file imports work on worker nodes in multi-node jobs."""
import ray
from data_processor import core

# Verify head node can use the import
assert core.process_number(10) is not None, "Head node import failed"


@ray.remote
def test_import_on_worker() -> None:
    from data_processor import core as worker_core

    # Verify import works on worker and return a computed value
    result = worker_core.process_number(10)
    assert result is not None, "Worker node import failed"


# Run tasks on workers to verify imports work
ray.get([test_import_on_worker.remote() for _ in range(10)])

"""Test that Python module imports work on worker nodes in multi-node jobs."""
import ray
from src.subdir.utils import tool

# Verify head node can use the import
assert tool.say_hi is not None, "Head node import failed"


@ray.remote
def test_import_on_worker() -> None:
    from src.subdir.utils import tool as worker_tool

    # Verify import works on worker and return the file path
    assert worker_tool.say_hi is not None, "Worker node import failed"
    assert worker_tool.__file__ is not None, "Worker tool has no __file__"


# Run tasks on workers to verify imports work
ray.get([test_import_on_worker.remote() for _ in range(10)])

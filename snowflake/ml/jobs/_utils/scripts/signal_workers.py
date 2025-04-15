#!/usr/bin/env python3
# This file is part of the Ray-based distributed job system for Snowflake ML.
# Architecture overview:
# - Head node creates a ShutdownSignal actor and signals workers when job completes
# - Worker nodes listen for this signal and gracefully shut down
# - This ensures clean termination of distributed Ray jobs
import argparse
import logging
import socket
import sys
import time
from typing import Any

import ray
from constants import (
    SHUTDOWN_ACTOR_NAME,
    SHUTDOWN_ACTOR_NAMESPACE,
    SHUTDOWN_RPC_TIMEOUT_SECONDS,
)
from ray.actor import ActorHandle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@ray.remote
class ShutdownSignal:
    """A simple Ray actor that workers can check to determine if they should shutdown"""

    def __init__(self) -> None:
        self.shutdown_requested = False
        self.timestamp = None
        self.hostname = socket.gethostname()
        self.acknowledged_workers = set()
        logging.info(f"ShutdownSignal actor created on {self.hostname}")

    def request_shutdown(self) -> dict[str, Any]:
        """Signal workers to shut down"""
        self.shutdown_requested = True
        self.timestamp = time.time()
        logging.info(f"Shutdown requested by head node at {self.timestamp}")
        return {"status": "shutdown_requested", "timestamp": self.timestamp, "host": self.hostname}

    def should_shutdown(self) -> dict[str, Any]:
        """Check if shutdown has been requested"""
        return {"shutdown": self.shutdown_requested, "timestamp": self.timestamp, "host": self.hostname}

    def ping(self) -> dict[str, Any]:
        """Simple method to test connectivity"""
        return {"status": "alive", "host": self.hostname}

    def acknowledge_shutdown(self, worker_id: str) -> dict[str, Any]:
        """Worker acknowledges it has received the shutdown signal and is terminating"""
        self.acknowledged_workers.add(worker_id)
        logging.info(f"Worker {worker_id} acknowledged shutdown. Total acknowledged: {len(self.acknowledged_workers)}")

        return {"status": "acknowledged", "worker_id": worker_id, "acknowledged_count": len(self.acknowledged_workers)}

    def get_acknowledgment_workers(self) -> set[str]:
        """Get the set of workers who have acknowledged shutdown"""
        return self.acknowledged_workers


def get_worker_node_ids() -> list[str]:
    """Get the IDs of all active worker nodes.

    Returns:
        List[str]: List of worker node IDs. Empty list if no workers are present.
    """
    worker_nodes = [
        node for node in ray.nodes() if node.get("Alive") and node.get("Resources", {}).get("node_tag:worker", 0) > 0
    ]

    worker_node_ids = [node.get("NodeName") for node in worker_nodes]

    if worker_node_ids:
        logging.info(f"Found {len(worker_node_ids)} worker nodes")
    else:
        logging.info("No active worker nodes found")

    return worker_node_ids


def get_or_create_shutdown_signal() -> ActorHandle:
    """Get existing shutdown signal actor or create a new one.

    Returns:
        ActorHandle: Reference to shutdown signal actor
    """
    try:
        # Try to get existing actor
        shutdown_signal = ray.get_actor(SHUTDOWN_ACTOR_NAME, namespace=SHUTDOWN_ACTOR_NAMESPACE)
        logging.info("Found existing shutdown signal actor")
    except (ValueError, ray.exceptions.RayActorError) as e:
        logging.info(f"Creating new shutdown signal actor: {e}")
        # Create new actor if it doesn't exist
        shutdown_signal = ShutdownSignal.options(
            name=SHUTDOWN_ACTOR_NAME,
            namespace=SHUTDOWN_ACTOR_NAMESPACE,
            lifetime="detached",  # Ensure actor survives client disconnect
            resources={"node_tag:head": 0.001},  # Resource constraint to ensure it runs on head node
        ).remote()

        # Verify actor is created and accessible
        ping_result = ray.get(shutdown_signal.ping.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS)
        logging.debug(f"New actor ping response: {ping_result}")

    return shutdown_signal


def request_shutdown(shutdown_signal: ActorHandle) -> None:
    """Request workers to shut down.

    Args:
        shutdown_signal: Reference to the shutdown signal actor
    """
    response = ray.get(shutdown_signal.request_shutdown.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS)
    logging.info(f"Shutdown requested: {response}")


def verify_shutdown(shutdown_signal: ActorHandle) -> None:
    """Verify that shutdown was properly signaled.

    Args:
        shutdown_signal: Reference to the shutdown signal actor
    """
    check = ray.get(shutdown_signal.should_shutdown.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS)
    logging.debug(f"Shutdown status check: {check}")


def wait_for_acknowledgments(shutdown_signal: ActorHandle, worker_node_ids: list[str], wait_time: int) -> None:
    """Wait for workers to acknowledge shutdown.

    Args:
        shutdown_signal: Reference to the shutdown signal actor
        worker_node_ids: List of worker node IDs
        wait_time: Time in seconds to wait for acknowledgments

    Raises:
        TimeoutError: When workers don't acknowledge within the wait time or if actor communication times out
    """
    if not worker_node_ids:
        return

    logging.info(f"Waiting up to {wait_time}s for workers to acknowledge shutdown signal...")
    start_time = time.time()
    check_interval = 1.0

    while time.time() - start_time < wait_time:
        try:
            ack_workers = ray.get(
                shutdown_signal.get_acknowledgment_workers.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS
            )
            if ack_workers and ack_workers == set(worker_node_ids):
                logging.info(
                    f"All {len(worker_node_ids)} workers acknowledged shutdown. "
                    f"Completed in {time.time() - start_time:.2f}s"
                )
                return
            else:
                logging.debug(f"Waiting for acknowledgments: {len(ack_workers)}/{len(worker_node_ids)} workers")
        except Exception as e:
            logging.warning(f"Error checking acknowledgment status: {e}")

        time.sleep(check_interval)

    raise TimeoutError(
        f"Timed out waiting for {len(worker_node_ids)} workers to acknowledge shutdown after {wait_time}s"
    )


def signal_workers(wait_time: int = 10) -> int:
    """
    Signal worker nodes to shut down by creating a shutdown signal actor.

    Args:
        wait_time: Time in seconds to wait for workers to receive the message

    Returns:
        0 for success, 1 for failure
    """
    ray.init(address="auto", ignore_reinit_error=True)

    worker_node_ids = get_worker_node_ids()

    if worker_node_ids:
        shutdown_signal = get_or_create_shutdown_signal()
        request_shutdown(shutdown_signal)
        verify_shutdown(shutdown_signal)
        wait_for_acknowledgments(shutdown_signal, worker_node_ids, wait_time)
    else:
        logging.info("No active worker nodes found to signal.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Ray workers to shutdown")
    parser.add_argument(
        "--wait-time", type=int, default=10, help="Time in seconds to wait for workers to receive the signal"
    )
    args = parser.parse_args()

    sys.exit(signal_workers(args.wait_time))

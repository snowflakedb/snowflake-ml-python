#!/usr/bin/env python3
# This file is part of the Ray-based distributed job system for Snowflake ML.
# Architecture overview:
# - Head node creates a ShutdownSignal actor and signals workers when job completes
# - Worker nodes listen for this signal via this script and gracefully shut down
# - This ensures clean termination of distributed Ray jobs
import logging
import signal
import sys
import time
from typing import Optional

import get_instance_ip
import ray
from constants import (
    SHUTDOWN_ACTOR_NAME,
    SHUTDOWN_ACTOR_NAMESPACE,
    SHUTDOWN_RPC_TIMEOUT_SECONDS,
)
from ray.actor import ActorHandle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_shutdown_actor() -> Optional[ActorHandle]:
    """
    Retrieve the shutdown signal actor from Ray.

    Returns:
        The shutdown signal actor or None if not found
    """
    try:
        shutdown_signal = ray.get_actor(SHUTDOWN_ACTOR_NAME, namespace=SHUTDOWN_ACTOR_NAMESPACE)
        return shutdown_signal
    except Exception:
        return None


def ping_shutdown_actor(shutdown_signal: ActorHandle) -> bool:
    """
    Ping the shutdown actor to ensure connectivity.

    Args:
        shutdown_signal: The Ray actor handle for the shutdown signal

    Returns:
        True if ping succeeds, False otherwise
    """
    try:
        ping_result = ray.get(shutdown_signal.ping.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS)
        logging.debug(f"Actor ping result: {ping_result}")
        return True
    except (ray.exceptions.GetTimeoutError, Exception) as e:
        logging.debug(f"Actor ping failed: {e}")
        return False


def check_shutdown_status(shutdown_signal: ActorHandle, worker_id: str) -> bool:
    """
    Check if worker should shutdown and acknowledge if needed.

    Args:
        shutdown_signal: The Ray actor handle for the shutdown signal
        worker_id: Worker identifier (IP address)

    Returns:
        True if should shutdown, False otherwise
    """
    try:
        status = ray.get(shutdown_signal.should_shutdown.remote(), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS)
        logging.debug(f"Shutdown status: {status}")

        if status.get("shutdown", False):
            logging.info(
                f"Received shutdown signal from head node at {status.get('timestamp')}. " f"Exiting worker process."
            )

            # Acknowledge shutdown before exiting
            try:
                ack_result = ray.get(
                    shutdown_signal.acknowledge_shutdown.remote(worker_id), timeout=SHUTDOWN_RPC_TIMEOUT_SECONDS
                )
                logging.info(f"Acknowledged shutdown: {ack_result}")
            except Exception as e:
                logging.warning(f"Failed to acknowledge shutdown: {e}. Continue to exit worker.")

            return True
        return False

    except Exception as e:
        logging.debug(f"Error checking shutdown status: {e}")
        return False


def check_ray_connectivity() -> bool:
    """
    Check if the Ray cluster is accessible.

    Returns:
        True if Ray is connected, False otherwise
    """
    try:
        # A simple check to verify Ray is working
        nodes = ray.nodes()
        if nodes:
            return True
        return False
    except Exception as e:
        logging.debug(f"Ray connectivity check failed: {e}")
        return False


def initialize_ray_connection(max_retries: int, initial_retry_delay: int, max_retry_delay: int) -> bool:
    """
    Initialize connection to Ray with retries.

    Args:
        max_retries: Maximum number of connection attempts
        initial_retry_delay: Initial delay between retries in seconds
        max_retry_delay: Maximum delay between retries in seconds

    Returns:
        bool: True if connection successful, False otherwise
    """
    retry_count = 0
    retry_delay = initial_retry_delay

    while retry_count < max_retries:
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            return True
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to connect to Ray head after {max_retries} attempts: {e}")
                return False

            logging.debug(
                f"Attempt {retry_count}/{max_retries} to connect to Ray failed: {e}. "
                f"Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
            # Exponential backoff with cap
            retry_delay = min(retry_delay * 1.5, max_retry_delay)

    return False  # Should not reach here, but added for completeness


def monitor_shutdown_signal(check_interval: int, max_consecutive_failures: int) -> int:
    """
    Main loop to monitor for shutdown signals.

    Args:
        check_interval: Time in seconds between checks
        max_consecutive_failures: Maximum allowed consecutive connection failures

    Returns:
        int: Exit code (0 for success, non-zero for failure)

    Raises:
        ConnectionError: If Ray connection failures exceed threshold
    """
    worker_id = get_instance_ip.get_self_ip()
    actor_check_count = 0
    consecutive_connection_failures = 0

    logging.debug(
        f"Starting to monitor for shutdown signal using actor {SHUTDOWN_ACTOR_NAME}"
        f" in namespace {SHUTDOWN_ACTOR_NAMESPACE}."
    )

    while True:
        actor_check_count += 1

        # Check Ray connectivity before proceeding
        if not check_ray_connectivity():
            consecutive_connection_failures += 1
            logging.debug(
                f"Ray connectivity check failed (attempt {consecutive_connection_failures}/{max_consecutive_failures})"
            )
            if consecutive_connection_failures >= max_consecutive_failures:
                raise ConnectionError("Exceeded max consecutive Ray connection failures")
            time.sleep(check_interval)
            continue

        # Reset counter on successful connection
        consecutive_connection_failures = 0

        # Get shutdown actor
        shutdown_signal = get_shutdown_actor()
        if not shutdown_signal:
            logging.debug(f"Shutdown signal actor not found at check #{actor_check_count}, continuing to wait...")
            time.sleep(check_interval)
            continue

        # Ping the actor to ensure connectivity
        if not ping_shutdown_actor(shutdown_signal):
            time.sleep(check_interval)
            continue

        # Check shutdown status
        if check_shutdown_status(shutdown_signal, worker_id):
            return 0

        # Wait before checking again
        time.sleep(check_interval)


def run_listener() -> int:
    """Listen for shutdown signals from the head node"""
    # Configuration
    max_retries = 15
    initial_retry_delay = 2
    max_retry_delay = 30
    check_interval = 5  # How often to check for ray connection or shutdown signal
    max_consecutive_failures = 12  # Exit after about 1 minute of connection failures

    # Initialize Ray connection
    if not initialize_ray_connection(max_retries, initial_retry_delay, max_retry_delay):
        raise ConnectionError("Failed to connect to Ray cluster. Aborting worker.")

    # Monitor for shutdown signals
    return monitor_shutdown_signal(check_interval, max_consecutive_failures)


def main():
    """Main entry point with signal handling"""

    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, exiting worker process.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run the listener - this will block until a shutdown signal is received
    result = run_listener()
    sys.exit(result)


if __name__ == "__main__":
    main()

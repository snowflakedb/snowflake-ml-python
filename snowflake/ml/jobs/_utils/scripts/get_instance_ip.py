#!/usr/bin/env python3
# This file is modified from mlruntime/service/snowflake/runtime/utils
import argparse
import logging
import socket
import sys
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_self_ip() -> Optional[str]:
    """Get the IP address of the current service instance.
    References:
    - https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-services#general-guidelines-related-to-service-to-service-communications # noqa: E501

    Returns:
        Optional[str]: The IP address of the current service instance, or None if unable to retrieve.
    """
    try:
        hostname = socket.gethostname()
        instance_ip = socket.gethostbyname(hostname)
        return instance_ip
    except OSError as e:
        logger.error(f"Error: Unable to get IP address via socket. {e}")
        return None


def get_first_instance(service_name: str) -> Optional[tuple[str, str, str]]:
    """Get the first instance of a batch job based on start time and instance ID.

    Args:
        service_name (str): The name of the service to query.

    Returns:
        tuple[str, str]: A tuple containing (instance_id, ip_address) of the head instance.
    """
    from snowflake.runtime.utils import session_utils

    session = session_utils.get_session()
    result = session.sql(f"show service instances in service {service_name}").collect()

    if not result:
        return None
    # we have already integrated with first_instance startup policy,
    # the instance 0 is guaranteed to be the head instance
    head_instance = next(
        (
            row
            for row in result
            if "instance_id" in row and row["instance_id"] is not None and int(row["instance_id"]) == 0
        ),
        None,
    )
    # fallback to find the first instance if the instance 0 is not found
    if not head_instance:
        # Sort by start_time first, then by instance_id. If start_time is null/empty, it will be sorted to the end.
        sorted_instances = sorted(
            result, key=lambda x: (not bool(x["start_time"]), x["start_time"], int(x["instance_id"]))
        )
        head_instance = sorted_instances[0]
    if not head_instance["instance_id"] or not head_instance["ip_address"]:
        return None
    # Validate head instance IP
    ip_address = head_instance["ip_address"]
    try:
        socket.inet_aton(ip_address)  # Validate IPv4 address
        return (head_instance["instance_id"], ip_address, head_instance["status"])
    except OSError:
        logger.error(f"Error: Invalid IP address format: {ip_address}")
        return None


def main():
    """Retrieves the IP address of a specified service instance or the current service.
    Args:
        service_name (str,required) Name of the service to query
        --instance-index (int, optional) Index of the service instance to query. Default: -1
            Currently only supports -1 to get the IP address of the current service instance.
        --head (bool, optional) Get the head instance information using show services.
            If set, instance-index will be ignored, and the script will return the index and IP address of
              the head instance, split by a space. Default: False.
        --timeout (int, optional) Maximum time to wait for IP address retrieval in seconds. Default: 720 seconds
        --retry-interval (int, optional) Time to wait between retry attempts in seconds. Default: 10 seconds
    Usage Examples:
        python get_instance_ip.py myservice --instance-index=1 --retry-interval=5
    Returns:
        Prints the IP address to stdout if successful. Exits with status code 0 on success, 1 on failure
    """

    parser = argparse.ArgumentParser(description="Get IP address of a service instance")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("service_name", help="Name of the service")
    group.add_argument(
        "--instance-index",
        type=int,
        default=-1,
        help="Index of service instance (default: -1 for self instance)",
    )
    group.add_argument(
        "--head",
        action="store_true",
        help="Get head instance information using show services",
    )
    parser.add_argument("--timeout", type=int, default=720, help="Timeout in seconds (default: 720)")
    parser.add_argument(
        "--retry-interval",
        type=int,
        default=10,
        help="Retry interval in seconds (default: 10)",
    )

    args = parser.parse_args()
    start_time = time.time()

    if args.head:
        while time.time() - start_time < args.timeout:
            head_info = get_first_instance(args.service_name)
            if head_info:
                # Print to stdout to allow capture but don't use logger
                sys.stdout.write(" ".join(head_info) + "\n")
                sys.exit(0)
            time.sleep(args.retry_interval)
        # If we get here, we've timed out
        logger.error("Error: Unable to retrieve head IP address")
        sys.exit(1)

    # If the index is -1, use get_self_ip to get the IP address of the current service
    if args.instance_index == -1:
        ip_address = get_self_ip()
        if ip_address:
            sys.stdout.write(f"{ip_address}\n")
            sys.exit(0)
        else:
            logger.error("Error: Unable to retrieve self IP address")
            sys.exit(1)
    else:
        # We don't support querying a specific instance index other than -1
        logger.error("Error: Invalid arguments. Only --instance-index=-1 is supported for now.")
        sys.exit(1)


if __name__ == "__main__":
    main()

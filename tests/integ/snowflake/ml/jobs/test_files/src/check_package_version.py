if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("package_spec", type=str, help="Package specifier like 'cloudpickle~=2.0'")
    args = parser.parse_args()

    # Parse package name from spec (e.g., "cloudpickle~=2.0" -> "cloudpickle")
    match = re.match(r"([a-zA-Z0-9_-]+)", args.package_spec)
    if not match:
        raise ValueError(f"Invalid package specifier: {args.package_spec}")
    package_name = match.group(1)

    module = __import__(package_name)
    version = getattr(module, "__version__", getattr(module, "version", "unknown"))
    print(f"{package_name} version: {version}")

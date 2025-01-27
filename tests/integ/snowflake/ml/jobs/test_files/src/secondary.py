import os

from main import job_fn

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_var_name", type=str, default="ENV_VAR")
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()

    print("Running secondary.py")
    job_fn(os.getenv(args.env_var_name), args.delay)

from datetime import datetime
from time import sleep


def job_fn(arg1: str, delay: float = 0) -> None:
    print(f"{datetime.now()}\t[{arg1}] Job start", flush=True)
    if delay > 0:
        sleep(delay)
    print(f"{datetime.now()}\t[{arg1}] Job complete", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("arg1", type=str)
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()

    print("Running main.py")
    job_fn(args.arg1, args.delay)

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
    parser.add_argument("--flag", type=str, required=False)
    args = parser.parse_args()
    print("Running main.py")
    print(f"arg1: {args.arg1}, delay: {args.delay}")
    if args.flag is not None and args.flag == "None":
        print("flag is None")
    job_fn(args.arg1, args.delay)

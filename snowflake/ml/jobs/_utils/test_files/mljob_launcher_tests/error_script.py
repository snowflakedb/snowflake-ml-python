# Script that raises an exception


def error_function():
    raise ValueError("Test error from function")


if __name__ == "__main__":
    # This will be executed when the script is run directly
    raise RuntimeError("Test error from script")

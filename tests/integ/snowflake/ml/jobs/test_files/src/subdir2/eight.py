from subdir3 import seven


def hello_from_six():
    return f"Six calling seven: [{seven.hello_from_seven()}]"


if __name__ == "__main__":
    print(hello_from_six())

VERSION = "1.0"


def say_hello(name: str) -> str:
    return f"Hello, {name}! Welcome from the greeter module (v{VERSION})."


def say_goodbye(name: str) -> str:
    return f"Goodbye, {name}!"

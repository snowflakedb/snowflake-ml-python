# Script that returns non-JSON-serializable objects


class CustomObject:
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self):
        return f"CustomObject({self.value})"


custom_obj = CustomObject(42)
complex_result = {"number": 42, "text": "hello", "custom": custom_obj, "function": lambda x: x * 2}

__return__ = complex_result

from custom_object_type import CustomObject

# Script that returns non-JSON-serializable objects
custom_obj = CustomObject(42)
complex_result = {"number": 42, "text": "hello", "custom": custom_obj, "function": lambda x: x * 2}

__return__ = complex_result

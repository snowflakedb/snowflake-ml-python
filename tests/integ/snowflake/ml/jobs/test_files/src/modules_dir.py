from src.utils import math_helpers as math, string_helpers
from src.utils.string_helpers import uppercase

reversed_name = string_helpers.reverse("Alice")
print(f"Option 1: {reversed_name}")

sum_result = math.add(10, 5)
print(f"Option 2: The sum is {sum_result}")
print(f"Option 2: Pi is {math.PI}")

upper_name = uppercase("bob")
print(f"Option 3: {upper_name}")

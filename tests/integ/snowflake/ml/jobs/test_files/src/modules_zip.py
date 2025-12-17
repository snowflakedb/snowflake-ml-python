from data_processor import core

# 3. Use the function from the 'core' module.
#    This works because 'core' is able to find and import 'validator'
#    from within the same zip archive.
print(f"Processing a valid number: {core.process_number(10)}")
print(f"Processing an invalid number: {core.process_number(-5)}")

# You can inspect the 'core' module's origin
print(f"\nThe 'core' module was loaded from: {core.__file__}")

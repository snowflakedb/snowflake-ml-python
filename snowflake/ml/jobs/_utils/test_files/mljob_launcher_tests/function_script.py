# Script with functions that can be called as main entry points


def main_function():
    return {"status": "success from function", "value": 100}


def another_function(value=0):
    return {"status": "success from another function", "value": value}


# This should be ignored when main_func is specified
__return__ = "This should be ignored"

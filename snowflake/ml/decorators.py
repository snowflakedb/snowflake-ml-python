import functools


def udf(func):
    @functools.wraps(func)
    def wrapper_udf(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_udf

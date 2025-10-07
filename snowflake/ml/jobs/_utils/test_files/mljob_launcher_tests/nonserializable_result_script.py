import threading


class NonserializableClass:
    def __init__(self, value) -> None:
        self.value = value
        self._lock = threading.Lock()  # Non-serializable attribute

    def __str__(self) -> str:
        return str(self.value)


__return__ = NonserializableClass(100)

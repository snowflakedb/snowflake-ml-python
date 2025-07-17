import importlib
from typing import Any, Generic, Type, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import typing_extensions as te

T = TypeVar("T")


class LazyType(Generic[T]):
    """Utility type to help defer need of importing."""

    def __init__(self, klass: Union[str, type[T]]) -> None:
        self.qualname = ""
        if isinstance(klass, str):
            parts = klass.rsplit(".", 1)
            assert len(parts) == 2, "Not a class"
            self.module, self.qualname = parts
            self._runtime_class = None
        else:
            self._runtime_class = klass
            self.module = klass.__module__
            if hasattr(klass, "__qualname__"):
                self.qualname = klass.__qualname__
            else:
                self.qualname = klass.__name__

    def __instancecheck__(self, obj: object) -> te.TypeGuard[T]:
        return self.isinstance(obj)

    @classmethod
    def from_type(cls, typ_: Union["LazyType[T]", type[T]]) -> "LazyType[T]":
        if isinstance(typ_, LazyType):
            return typ_
        return cls(typ_)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type):
            o = self.__class__(o)
        if isinstance(o, LazyType):
            return self.module == o.module and self.qualname == o.qualname
        return False

    def __hash__(self) -> int:
        return hash(f"{self.module}.{self.qualname}")

    def __repr__(self) -> str:
        return f'LazyType("{self.module}", "{self.qualname}")'

    def get_class(self) -> type[T]:
        if self._runtime_class is None:
            try:
                m = importlib.import_module(self.module)
            except ModuleNotFoundError:
                raise ValueError(f"Module {self.module} not imported.")

            self._runtime_class = cast("Type[T]", getattr(m, self.qualname))

        return self._runtime_class

    def isinstance(self, obj: Any) -> te.TypeGuard[T]:
        try:
            return isinstance(obj, self.get_class())
        except ValueError:
            return False


LiteralNDArrayType = Union[npt.NDArray[np.int_], npt.NDArray[np.float64], npt.NDArray[np.str_], npt.NDArray[np.bool_]]

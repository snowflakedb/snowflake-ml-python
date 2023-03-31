import functools
import inspect
from abc import ABC
from typing import Any, Callable, Dict, Optional

import anyio

from snowflake.ml.model import type_spec

ModelType = Any


class MethodRef:
    """Represents an method invocation of an instance of `ModelRef`.

    This allows us to
        1) Customize the place of actual execution of the method(inline, thread/process pool or remote).
        2) Enrich the way of execution(sync versus async).
    Example:
        If you have a SKL model, you would normally invoke by `skl_ref.predict(df)` which has sync API.
        Within inference graph, you could invoke `await skl_refa.predict.async_run(df)` which automatically
        will be run on thread with async interface.
    """

    def __init__(self, model_ref: "ModelRef", method_name: str) -> None:
        self._func = getattr(model_ref.model, method_name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    async def async_run(self, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self._func):
            return await self._func(*args, **kwargs)
        return await anyio.to_thread.run_sync(functools.partial(self._func, **kwargs), *args)


class ModelRef:
    """Represents an model in the inference graph.

    This enables us to separate physical and logical representation of a model which
    will allows us to deeply understand the graph and perform optimization at entire
    graph level.
    """

    def __init__(self, name: str, model: ModelType) -> None:
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __getattr__(self, method_name: str) -> Any:
        if hasattr(self._model, method_name):
            return MethodRef(self, method_name)
        else:
            self.__getattribute__(method_name)

    def __getstate__(self) -> Any:
        state = self.__dict__.copy()
        del state["_model"]
        return state

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)

    @property
    def model(self) -> ModelType:
        return self._model


class ModelContext:
    def __init__(
        self,
        *,
        artifacts: Optional[Dict[str, str]] = None,
        models: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.artifacts = artifacts if artifacts else dict()
        self.model_refs = {name: ModelRef(name, model) for name, model in models.items()} if models else dict()

    def path(self, key: str) -> str:
        return self.artifacts[key]

    def model_ref(self, name: str) -> ModelRef:
        return self.model_refs[name]


class CustomModel(ABC):
    _input_spec: Optional[type_spec.TypeSpec] = None
    _output_spec: Optional[type_spec.TypeSpec] = None
    _user_func: Optional[Callable[..., Any]] = None

    def __init__(self, context: ModelContext) -> None:
        self.context = context

    # TODO(halu): Fix decorator typing. To constraint type further to wrapped function.
    @classmethod
    def api(
        cls,
        input_spec: type_spec.TypeSpec,
        output_spec: type_spec.TypeSpec,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def inner(user_func: Callable[..., Any]) -> Callable[..., Any]:
            cls._user_func = user_func
            cls._input_spec = input_spec
            cls._output_spec = output_spec

            @functools.wraps(user_func)
            def wrapper(self: "CustomModel", arg: Any) -> Any:
                return user_func(self, arg)

            return wrapper

        return inner

    def _validate(self) -> None:
        if not self._input_spec or not self._output_spec:
            raise RuntimeError("Input or output spec not specified.")
        assert self._user_func is not None
        sig = inspect.signature(self._user_func)
        assert len(sig.parameters) == 1, "API should only has single argument."
        assert (
            sig.parameters[next(iter(sig.parameters))].annotation == self._input_spec.py_type()
        ), f"Input type mis-matching. spec: f{self._input_spec.py_type()},"
        " annotation: f{sig.parameters[next(iter(sig.parameters))].annotation}"
        assert sig.return_annotation == self._output_spec.py_type(), "output type mis-match."

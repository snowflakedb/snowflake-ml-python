import inspect
from typing import Any, Callable, Optional

from snowflake import snowpark
from snowflake.snowpark import context as sp_context


class FunctionPayload:
    def __init__(
        self,
        func: Callable[..., Any],
        session: Optional[snowpark.Session] = None,
        session_argument: str = "",
        *args: Any,
        **kwargs: Any
    ) -> None:
        self.function = func
        self.args = args
        self.kwargs = kwargs
        self._session = session
        self._session_argument = session_argument

    @property
    def session(self) -> Optional[snowpark.Session]:
        return self._session

    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to exclude session."""
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore session from context during unpickling."""
        self.__dict__.update(state)
        self._session = sp_context.get_active_session()

    def __call__(self) -> Any:
        sig = inspect.signature(self.function)
        bound = sig.bind_partial(*self.args, **self.kwargs)
        bound.arguments[self._session_argument] = self._session

        return self.function(*bound.args, **bound.kwargs)

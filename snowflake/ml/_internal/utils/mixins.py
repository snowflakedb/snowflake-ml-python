from dataclasses import dataclass
from typing import Any, Optional

from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import session as snowpark_session

_SESSION_KEY = "_session"
_SESSION_ACCOUNT_KEY = "session$account"
_SESSION_ROLE_KEY = "session$role"
_SESSION_DATABASE_KEY = "session$database"
_SESSION_SCHEMA_KEY = "session$schema"
_SESSION_STATE_ATTR = "_session_state"


def _identifiers_match(saved: Optional[str], current: Optional[str]) -> bool:
    saved_resolved = identifier.resolve_identifier(saved) if saved is not None else saved
    current_resolved = identifier.resolve_identifier(current) if current is not None else current
    return saved_resolved == current_resolved


@dataclass(frozen=True)
class _SessionState:
    account: Optional[str]
    role: Optional[str]
    database: Optional[str]
    schema: Optional[str]


class SerializableSessionMixin:
    """Mixin that provides pickling capabilities for objects with Snowpark sessions."""

    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to exclude non-serializable session and related components."""
        parent_state = (
            super().__getstate__()  # type: ignore[misc] # object.__getstate__ appears in 3.11
            if hasattr(super(), "__getstate__")
            else self.__dict__
        )
        state = dict(parent_state)  # Create a copy so we can safely modify the state

        # Save session metadata for validation during unpickling
        session = state.pop(_SESSION_KEY, None)
        if session is not None:
            state[_SESSION_ACCOUNT_KEY] = session.get_current_account()
            state[_SESSION_ROLE_KEY] = session.get_current_role()
            state[_SESSION_DATABASE_KEY] = session.get_current_database()
            state[_SESSION_SCHEMA_KEY] = session.get_current_schema()

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore session from context during unpickling."""
        session_state = _SessionState(
            account=state.pop(_SESSION_ACCOUNT_KEY, None),
            role=state.pop(_SESSION_ROLE_KEY, None),
            database=state.pop(_SESSION_DATABASE_KEY, None),
            schema=state.pop(_SESSION_SCHEMA_KEY, None),
        )

        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # type: ignore[misc]
        else:
            self.__dict__.update(state)

        setattr(self, _SESSION_STATE_ATTR, session_state)

    def _set_session(self, session_state: _SessionState) -> None:

        if session_state.account is not None:
            active_sessions = snowpark_session._get_active_sessions()
            if len(active_sessions) == 0:
                raise RuntimeError("No active Snowpark session available. Please create a session.")

            # Best effort match: Find the session with the most matching identifiers
            setattr(
                self,
                _SESSION_KEY,
                max(
                    active_sessions,
                    key=lambda s: sum(
                        (
                            _identifiers_match(session_state.account, s.get_current_account()),
                            _identifiers_match(session_state.role, s.get_current_role()),
                            _identifiers_match(session_state.database, s.get_current_database()),
                            _identifiers_match(session_state.schema, s.get_current_schema()),
                        )
                    ),
                ),
            )

    @property
    def session(self) -> Optional[snowpark_session.Session]:
        if _SESSION_KEY not in self.__dict__:
            session_state = getattr(self, _SESSION_STATE_ATTR, None)
            if session_state is not None:
                self._set_session(session_state)
        return self.__dict__.get(_SESSION_KEY)

    @session.setter
    def session(self, value: Optional[snowpark_session.Session]) -> None:
        self.__dict__[_SESSION_KEY] = value

    # _getattr__ is only called when an attribute is NOT found through normal lookup.
    # 1. Data descriptors (like @property with setter) from the class hierarchy
    # 2. Instance __dict__ (e.g., self.x = 10)
    # 3. Non-data descriptors (methods, `@property without setter) from the class hierarchy
    # __getattr__ — only called if steps 1-3 all fail
    def __getattr__(self, name: str) -> Any:
        if name == _SESSION_KEY:
            return self.session
        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)  # type: ignore[misc]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

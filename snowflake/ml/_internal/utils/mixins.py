from typing import Any, Optional

from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import session as snowpark_session

_SESSION_KEY = "_session"
_SESSION_ACCOUNT_KEY = "session$account"
_SESSION_ROLE_KEY = "session$role"
_SESSION_DATABASE_KEY = "session$database"
_SESSION_SCHEMA_KEY = "session$schema"


def _identifiers_match(saved: Optional[str], current: Optional[str]) -> bool:
    saved_resolved = identifier.resolve_identifier(saved) if saved is not None else saved
    current_resolved = identifier.resolve_identifier(current) if current is not None else current
    return saved_resolved == current_resolved


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
        saved_account = state.pop(_SESSION_ACCOUNT_KEY, None)
        saved_role = state.pop(_SESSION_ROLE_KEY, None)
        saved_database = state.pop(_SESSION_DATABASE_KEY, None)
        saved_schema = state.pop(_SESSION_SCHEMA_KEY, None)

        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # type: ignore[misc]
        else:
            self.__dict__.update(state)

        if saved_account is not None:
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
                            _identifiers_match(saved_account, s.get_current_account()),
                            _identifiers_match(saved_role, s.get_current_role()),
                            _identifiers_match(saved_database, s.get_current_database()),
                            _identifiers_match(saved_schema, s.get_current_schema()),
                        )
                    ),
                ),
            )

from typing import Any, Optional

from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import session


class SerializableSessionMixin:
    """Mixin that provides pickling capabilities for objects with Snowpark sessions."""

    def __getstate__(self) -> dict[str, Any]:
        """Customize pickling to exclude non-serializable session and related components."""
        state = self.__dict__.copy()

        # Save session metadata for validation during unpickling
        if hasattr(self, "_session") and self._session is not None:
            try:
                state["__session-account__"] = self._session.get_current_account()
                state["__session-role__"] = self._session.get_current_role()
                state["__session-database__"] = self._session.get_current_database()
                state["__session-schema__"] = self._session.get_current_schema()
            except Exception:
                pass

        state["_session"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore session from context during unpickling."""
        saved_account = state.pop("__session-account__", None)
        saved_role = state.pop("__session-role__", None)
        saved_database = state.pop("__session-database__", None)
        saved_schema = state.pop("__session-schema__", None)
        self.__dict__.update(state)

        if saved_account is not None:

            def identifiers_match(saved: Optional[str], current: Optional[str]) -> bool:
                saved_resolved = identifier.resolve_identifier(saved) if saved is not None else saved
                current_resolved = identifier.resolve_identifier(current) if current is not None else current
                return saved_resolved == current_resolved

            for active_session in session._get_active_sessions():
                try:
                    current_account = active_session.get_current_account()
                    current_role = active_session.get_current_role()
                    current_database = active_session.get_current_database()
                    current_schema = active_session.get_current_schema()

                    if (
                        identifiers_match(saved_account, current_account)
                        and identifiers_match(saved_role, current_role)
                        and identifiers_match(saved_database, current_database)
                        and identifiers_match(saved_schema, current_schema)
                    ):
                        self._session = active_session
                        return
                except Exception:
                    continue

        # No matching session found or no metadata available
        raise RuntimeError("No active Snowpark session available. Please create a session.")

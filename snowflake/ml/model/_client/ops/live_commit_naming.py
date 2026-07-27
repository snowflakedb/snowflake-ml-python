from uuid import uuid4

from snowflake.ml._internal.utils import sql_identifier

_CHECKOUT_NAME_SUFFIX_LEN = 8


def _checkout_name_suffix() -> str:
    return uuid4().hex[:_CHECKOUT_NAME_SUFFIX_LEN].upper()


def generate_pending_model_name() -> sql_identifier.SqlIdentifier:
    """Generate a hidden pending model name for checkout (CREATE path)."""
    return sql_identifier.SqlIdentifier(f"PENDING_{_checkout_name_suffix()}_MODEL")


def generate_live_version_name() -> sql_identifier.SqlIdentifier:
    """Generate a hidden live version name for checkout."""
    return sql_identifier.SqlIdentifier(f"LIVE_{_checkout_name_suffix()}_VERSION")

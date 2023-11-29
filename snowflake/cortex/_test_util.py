import signal
from typing import Any, cast

from snowflake import snowpark
from snowflake.ml.utils import connection_params


def create_test_session() -> snowpark.Session:
    """Creates a Snowflake session under a timeout."""

    def handle_timeout(_signum: Any, _frame: Any) -> None:
        raise Exception("Timed out creating snowflake session. VPN connection may be required.")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(30)  # 30s timeout.
    session = snowpark.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
    signal.alarm(0)
    return cast(snowpark.Session, session)

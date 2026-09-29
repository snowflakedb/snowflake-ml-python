from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities
from snowflake.snowpark import session


def skip_unless_lora_adapters_account_enabled(sess: session.Session) -> None:
    """Skip when ENABLE_LORA_ADAPTERS is off or absent on this account.

    Reads the account parameter, not the session. A session SET would hide the
    account default and run these tests where the feature is still off.

    Args:
        sess: Active Snowpark session used to read the account parameter.

    Raises:
        SkipTest: If the account parameter is missing or not true.
    """
    rows = sess.sql(f"SHOW PARAMETERS LIKE '{platform_capabilities.ENABLE_LORA_ADAPTERS}' IN ACCOUNT").collect()
    if not rows or str(rows[0]["value"]).lower() != "true":
        raise absltest.SkipTest("ENABLE_LORA_ADAPTERS is not enabled on this account")

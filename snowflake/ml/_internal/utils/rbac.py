from snowflake import snowpark

# Constants for privileges.

PRIVILEGE_APPLY_MASKING_POLICY = "APPLY MASKING POLICY"
PRIVILEGE_APPLY_PASSWORD_POLICY = "APPLY PASSWORD POLICY"
PRIVILEGE_APPLY_SESSION_POLICY = "APPLY SESSION POLICY"
PRIVILEGE_ATTACH_POLICY = "ATTACH POLICY"
PRIVILEGE_CREATE_ACCOUNT = "CREATE ACCOUNT"
PRIVILEGE_CREATE_CREDENTIAL = "CREATE CREDENTIAL"
PRIVILEGE_CREATE_DATA_EXCHANGE_LISTING = "CREATE DATA EXCHANGE LISTING"
PRIVILEGE_CREATE_DATABASE = "CREATE DATABASE"
PRIVILEGE_CREATE_FAILOVER_GROUP = "CREATE FAILOVER GROUP"
PRIVILEGE_CREATE_INTEGRATION = "CREATE INTEGRATION"
PRIVILEGE_CREATE_NETWORK_POLICY = "CREATE NETWORK POLICY"
PRIVILEGE_CREATE_REPLICATION_GROUP = "CREATE REPLICATION GROUP"
PRIVILEGE_CREATE_ROLE = "CREATE ROLE"
PRIVILEGE_CREATE_SHARE = "CREATE SHARE"
PRIVILEGE_CREATE_USER = "CREATE USER"
PRIVILEGE_CREATE_WAREHOUSE = "CREATE WAREHOUSE"
PRIVILEGE_EXECUTE_MANAGED_TASK = "EXECUTE MANAGED TASK"
PRIVILEGE_EXECUTE_TASK = "EXECUTE TASK"
PRIVILEGE_IMPORT_SHARE = "IMPORT SHARE"
PRIVILEGE_MANAGE_ACCOUNT_SUPPORT_CASES = "MANAGE ACCOUNT SUPPORT CASES"
PRIVILEGE_MANAGE_GRANTS = "MANAGE GRANTS"
PRIVILEGE_MANAGE_ORGANIZATION_SUPPORT_CASES = "MANAGE ORGANIZATION SUPPORT CASES"
PRIVILEGE_MANAGE_USER_SUPPORT_CASES = "MANAGE USER SUPPORT CASES"
PRIVILEGE_MONITOR = "MONITOR"
PRIVILEGE_MONITOR_EXECUTION = "MONITOR EXECUTION"
PRIVILEGE_MONITOR_SECURITY = "MONITOR SECURITY"
PRIVILEGE_MONITOR_USAGE = "MONITOR USAGE"
PRIVILEGE_OVERRIDE_SHARE_RESTRICTIONS = "OVERRIDE SHARE RESTRICTIONS"
PRIVILEGE_PURCHASE_DATA_EXCHANGE_LISTING = "PURCHASE DATA EXCHANGE LISTING"


def get_role_privileges(session: snowpark.Session, role_name: str) -> set[str]:
    """Returns the set of privileges for the given role.

    Args:
        session (snowpark.Session): Authenticated snowpark session to communicate with the Snowflake backend.
        role_name (str): Name of the role to retrieve privileges for.

    Returns:
        The set of privileges for the given role.
    """
    # Remove quotes around the role name since session.get_current_role() adds double quotes to the string.
    sanitized_role_name = role_name.strip('"')

    # The result of SHOW GRANTS is not a regular dataframe and needs to be handled manually.
    def filter_predicate(x: snowpark.Row) -> bool:
        return bool(
            x["grant_option"] == "false" and x["granted_to"] == "ROLE" and x["grantee_name"] == sanitized_role_name
        )

    grants_on_account = session.sql("SHOW GRANTS ON ACCOUNT").collect()
    return {x["privilege"] for x in filter(filter_predicate, grants_on_account)}

import datetime
from typing import List, Optional
from uuid import uuid4

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier
from snowflake.ml.utils import sql_client

_COMMON_PREFIX = "snowml_test_"
_default_creation_mode = sql_client.CreationMode()


class DBManager:
    def __init__(self, session: snowpark.Session) -> None:
        self._session = session
        # prefix example: snowml_test_20230630120429_8e

    def set_role(self, role: str) -> None:
        self._session.sql(f"USE ROLE {role}").collect()

    def set_warehouse(self, warehouse: str) -> None:
        self._session.sql(f"USE WAREHOUSE {warehouse}").collect()

    def create_database(
        self,
        db_name: str,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
    ) -> str:
        actual_db_name = identifier.get_inferred_name(db_name)
        ddl_phrases = creation_mode.get_ddl_phrases()
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} DATABASE"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} "
            f"{actual_db_name} DATA_RETENTION_TIME_IN_DAYS = 0"
        ).collect()
        return actual_db_name

    def use_database(self, db_name: str) -> None:
        actual_db_name = identifier.get_inferred_name(db_name)
        self._session.sql(f"USE DATABASE {actual_db_name}").collect()

    def show_databases(self, db_name: str) -> snowpark.DataFrame:
        sql = f"SHOW DATABASES LIKE '{db_name}'"
        return self._session.sql(sql)

    def assert_database_existence(self, db_name: str, exists: bool = True) -> bool:
        count = self.show_databases(db_name).count()
        return count != 0 if exists else count == 0

    def drop_database(self, db_name: str, if_exists: bool = False) -> None:
        actual_db_name = identifier.get_inferred_name(db_name)
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP DATABASE{if_exists_sql} {actual_db_name}").collect()

    def cleanup_databases(self, expire_hours: int = 72) -> None:
        databases_df = self.show_databases(f"{_COMMON_PREFIX}%")
        stale_databases = databases_df.filter(
            f"\"created_on\" < dateadd('hour', {-expire_hours}, current_timestamp())"
        ).collect()
        for stale_db in stale_databases:
            self.drop_database(stale_db.name, if_exists=True)

    def create_schema(
        self,
        schema_name: str,
        db_name: Optional[str] = None,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
    ) -> str:
        actual_schema_name = identifier.get_inferred_name(schema_name)
        if db_name:
            actual_db_name = self.create_database(db_name, creation_mode=sql_client.CreationMode(if_not_exists=True))
            full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
        else:
            full_qual_schema_name = actual_schema_name
        ddl_phrases = creation_mode.get_ddl_phrases()
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} SCHEMA"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_schema_name}"
        ).collect()
        return full_qual_schema_name

    def create_random_schema(
        self,
        prefix: str = _COMMON_PREFIX,
        db_name: Optional[str] = None,
    ) -> str:
        schema_name = f"{prefix}_{uuid4().hex.upper()}"
        return self.create_schema(schema_name, db_name=db_name)

    def use_schema(
        self,
        schema_name: str,
        db_name: Optional[str] = None,
    ) -> None:
        actual_schema_name = identifier.get_inferred_name(schema_name)
        if db_name:
            actual_db_name = identifier.get_inferred_name(db_name)
            full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
        else:
            full_qual_schema_name = actual_schema_name
        self._session.sql(f"USE SCHEMA {full_qual_schema_name}").collect()

    def show_schemas(self, schema_name: str, db_name: Optional[str] = None) -> snowpark.DataFrame:
        if db_name:
            actual_db_name = identifier.get_inferred_name(db_name)
            location_sql = f" IN DATABASE {actual_db_name}"
        else:
            location_sql = ""
        sql = f"SHOW SCHEMAS LIKE '{schema_name}'{location_sql}"
        return self._session.sql(sql)

    def assert_schema_existence(self, schema_name: str, db_name: Optional[str] = None, exists: bool = False) -> bool:
        count = self.show_schemas(schema_name, db_name).count()
        return count != 0 if exists else count == 0

    def drop_schema(self, schema_name: str, db_name: Optional[str] = None, if_exists: bool = False) -> None:
        actual_schema_name = identifier.get_inferred_name(schema_name)
        if db_name:
            actual_db_name = identifier.get_inferred_name(db_name)
            full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
        else:
            full_qual_schema_name = actual_schema_name
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP SCHEMA{if_exists_sql} {full_qual_schema_name}").collect()

    def cleanup_schemas(
        self, prefix: str = _COMMON_PREFIX, db_name: Optional[str] = None, expire_days: int = 3
    ) -> None:
        schemas_df = self.show_schemas(f"{prefix}%", db_name)
        stale_schemas = schemas_df.filter(
            f"\"created_on\" < dateadd('day', {-expire_days}, current_timestamp())"
        ).collect()
        for stale_schema in stale_schemas:
            self.drop_schema(stale_schema.name, db_name, if_exists=True)

    def create_stage(
        self,
        stage_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
        sse_encrypted: bool = False,
    ) -> str:
        actual_stage_name = identifier.get_inferred_name(stage_name)
        if schema_name:
            full_qual_schema_name = self.create_schema(
                schema_name, db_name, creation_mode=sql_client.CreationMode(if_not_exists=True)
            )
            full_qual_stage_name = f"{full_qual_schema_name}.{actual_stage_name}"
        else:
            full_qual_stage_name = actual_stage_name
        ddl_phrases = creation_mode.get_ddl_phrases()
        encryption_sql = " ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')" if sse_encrypted else ""
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} STAGE"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_stage_name}{encryption_sql}"
        ).collect()
        return full_qual_stage_name

    @staticmethod
    def get_show_location_url(
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> str:
        if schema_name:
            actual_schema_name = identifier.get_inferred_name(schema_name)
            if db_name:
                actual_db_name = identifier.get_inferred_name(db_name)
                full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
            else:
                full_qual_schema_name = actual_schema_name
            location_sql = f" IN SCHEMA {full_qual_schema_name}"
        else:
            location_sql = ""
        return location_sql

    def show_stages(
        self,
        stage_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> snowpark.DataFrame:
        location_sql = DBManager.get_show_location_url(schema_name, db_name)
        sql = f"SHOW STAGES LIKE '{stage_name}'{location_sql}"
        return self._session.sql(sql)

    def assert_stage_existence(
        self, stage_name: str, schema_name: Optional[str] = None, db_name: Optional[str] = None, exists: bool = True
    ) -> bool:
        count = self.show_stages(stage_name, schema_name, db_name).count()
        return count != 0 if exists else count == 0

    def drop_stage(
        self, stage_name: str, schema_name: Optional[str] = None, db_name: Optional[str] = None, if_exists: bool = False
    ) -> None:
        actual_stage_name = identifier.get_inferred_name(stage_name)
        if schema_name:
            actual_schema_name = identifier.get_inferred_name(schema_name)
            if db_name:
                actual_db_name = identifier.get_inferred_name(db_name)
                full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
            else:
                full_qual_schema_name = actual_schema_name
            full_qual_stage_name = f"{full_qual_schema_name}.{actual_stage_name}"
        else:
            full_qual_stage_name = actual_stage_name
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP STAGE{if_exists_sql} {full_qual_stage_name}").collect()

    def cleanup_stages(
        self, schema_name: Optional[str] = None, db_name: Optional[str] = None, expire_days: int = 3
    ) -> None:
        stages_df = self.show_stages(f"{_COMMON_PREFIX}%", schema_name, db_name)
        stale_stages = stages_df.filter(
            f"\"created_on\" < dateadd('day', {-expire_days}, current_timestamp())"
        ).collect()
        for stale_stage in stale_stages:
            self.drop_stage(stale_stage.name, schema_name, db_name, if_exists=True)

    def show_user_functions(
        self,
        function_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> snowpark.DataFrame:
        location_sql = DBManager.get_show_location_url(schema_name, db_name)
        sql = f"SHOW USER FUNCTIONS LIKE '{function_name}'{location_sql}"
        return self._session.sql(sql)

    def drop_function(
        self,
        function_name: Optional[str] = None,
        args: Optional[List[str]] = None,
        function_def: Optional[str] = None,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        if_exists: bool = True,
    ) -> None:
        if not function_def:
            assert function_name is not None
            assert args is not None
            actual_function_name = identifier.get_inferred_name(function_name)
            args_str = f"({', '.join(args)})"
            function_def = actual_function_name + args_str

        if schema_name:
            actual_schema_name = identifier.get_inferred_name(schema_name)
            if db_name:
                actual_db_name = identifier.get_inferred_name(db_name)
                full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
            else:
                full_qual_schema_name = actual_schema_name
            full_qual_function_def = f"{full_qual_schema_name}.{function_def}"
        else:
            full_qual_function_def = function_def

        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP FUNCTION{if_exists_sql} {full_qual_function_def}").collect()

    def cleanup_user_functions(
        self, schema_name: Optional[str] = None, db_name: Optional[str] = None, expire_days: int = 3
    ) -> None:
        user_functions_df = self.show_user_functions(f"{_COMMON_PREFIX}%", schema_name, db_name)
        stale_funcs = user_functions_df.filter(
            f"\"created_on\" < dateadd('day', {-expire_days}, current_timestamp())"
        ).collect()
        for stale_func in stale_funcs:
            func_arguments = str(stale_func.arguments)
            func_def = func_arguments.partition("RETURN")[0].strip()
            self.drop_function(function_def=func_def, schema_name=schema_name, db_name=db_name, if_exists=True)

    def create_compute_pool(
        self,
        compute_pool_name: str,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
        instance_family: str = "CPU_X64_XS",
        min_nodes: int = 1,
        max_nodes: int = 1,
    ) -> str:
        full_qual_compute_pool_name = identifier.get_inferred_name(compute_pool_name)
        ddl_phrases = creation_mode.get_ddl_phrases()
        instance_family_sql = f" INSTANCE_FAMILY = '{instance_family}'"
        min_nodes_sql = f" MIN_NODES = {min_nodes}"
        max_nodes_sql = f" MAX_NODES = {max_nodes}"
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} COMPUTE POOL"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_compute_pool_name}"
            f"{instance_family_sql}{min_nodes_sql}{max_nodes_sql}"
        ).collect()
        return full_qual_compute_pool_name

    def drop_compute_pool(
        self,
        compute_pool_name: str,
        if_exists: bool = False,
    ) -> None:
        full_qual_compute_pool_name = identifier.get_inferred_name(compute_pool_name)
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP COMPUTE POOL{if_exists_sql} {full_qual_compute_pool_name}").collect()

    def create_image_repo(
        self,
        image_repo_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
    ) -> str:
        actual_image_repo_name = identifier.get_inferred_name(image_repo_name)
        if schema_name:
            full_qual_schema_name = self.create_schema(
                schema_name, db_name, creation_mode=sql_client.CreationMode(if_not_exists=True)
            )
            full_qual_image_repo_name = f"{full_qual_schema_name}.{actual_image_repo_name}"
        else:
            full_qual_image_repo_name = actual_image_repo_name
        ddl_phrases = creation_mode.get_ddl_phrases()
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} IMAGE REPOSITORY"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_image_repo_name}"
        ).collect()
        return full_qual_image_repo_name

    def drop_image_repo(
        self,
        image_repo_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        if_exists: bool = False,
    ) -> None:
        actual_image_repo_name = identifier.get_inferred_name(image_repo_name)
        if schema_name:
            actual_schema_name = identifier.get_inferred_name(schema_name)
            if db_name:
                actual_db_name = identifier.get_inferred_name(db_name)
                full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
            else:
                full_qual_schema_name = actual_schema_name
            full_qual_image_repo_name = f"{full_qual_schema_name}.{actual_image_repo_name}"
        else:
            full_qual_image_repo_name = actual_image_repo_name
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP IMAGE REPOSITORY{if_exists_sql} {full_qual_image_repo_name}").collect()

    def create_network_rule(
        self,
        network_rule_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
        mode: str = "EGRESS",
        type: str = "HOST_PORT",
        value_list: Optional[List[str]] = None,
    ) -> str:
        actual_network_rule_name = identifier.get_inferred_name(network_rule_name)
        if schema_name:
            full_qual_schema_name = self.create_schema(
                schema_name, db_name, creation_mode=sql_client.CreationMode(if_not_exists=True)
            )
            full_qual_network_rule_name = f"{full_qual_schema_name}.{actual_network_rule_name}"
        else:
            full_qual_network_rule_name = actual_network_rule_name
        ddl_phrases = creation_mode.get_ddl_phrases()
        mode_sql = f" MODE = '{mode}'"
        type_sql = f" TYPE = '{type}'"
        value_list = [] if value_list is None else value_list
        value_list_val = ", ".join([f"'{v}'" for v in value_list])
        value_list_sql = f" VALUE_LIST = ({value_list_val})"
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} NETWORK RULE"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_network_rule_name}"
            f"{mode_sql}{type_sql}{value_list_sql}"
        ).collect()
        return full_qual_network_rule_name

    def drop_network_rule(
        self,
        network_rule_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
        if_exists: bool = False,
    ) -> None:
        actual_network_rule_name = identifier.get_inferred_name(network_rule_name)
        if schema_name:
            actual_schema_name = identifier.get_inferred_name(schema_name)
            if db_name:
                actual_db_name = identifier.get_inferred_name(db_name)
                full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
            else:
                full_qual_schema_name = actual_schema_name
            full_qual_network_rule_name = f"{full_qual_schema_name}.{actual_network_rule_name}"
        else:
            full_qual_network_rule_name = actual_network_rule_name
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP NETWORK RULE{if_exists_sql} {full_qual_network_rule_name}").collect()

    def create_external_access_integration(
        self,
        external_access_integration_name: str,
        creation_mode: sql_client.CreationMode = _default_creation_mode,
        allowed_network_rules: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> str:
        full_qual_eai_name = identifier.get_inferred_name(external_access_integration_name)
        ddl_phrases = creation_mode.get_ddl_phrases()
        allowed_network_rules = [] if allowed_network_rules is None else allowed_network_rules
        allowed_network_rules_sql = f" ALLOWED_NETWORK_RULES = ({', '.join(allowed_network_rules)})"
        enabled_sql = f" ENABLED = {enabled}"
        self._session.sql(
            f"CREATE{ddl_phrases[sql_client.CreationOption.OR_REPLACE]} EXTERNAL ACCESS INTEGRATION"
            f"{ddl_phrases[sql_client.CreationOption.CREATE_IF_NOT_EXIST]} {full_qual_eai_name}"
            f"{allowed_network_rules_sql}{enabled_sql}"
        ).collect()
        return full_qual_eai_name

    def drop_external_access_integration(self, external_access_integration_name: str, if_exists: bool = False) -> None:
        full_qual_eai_name = identifier.get_inferred_name(external_access_integration_name)
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP EXTERNAL ACCESS INTEGRATION{if_exists_sql} {full_qual_eai_name}").collect()


class TestObjectNameGenerator:
    @staticmethod
    def get_snowml_test_object_name(run_id: str, suffix: str) -> str:
        """Generates a unified object name for naming all snowml non-session scoped artifacts.

        Args:
            run_id: Unique run id, suggesting to use uuid.uuid4().hex
            suffix: Custom suffix.

        Returns:
            An object name in a uniform pattern like "snowml_test_20230630120429_runid_suffix"
        """
        return f"{_COMMON_PREFIX}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{run_id}_{suffix}"

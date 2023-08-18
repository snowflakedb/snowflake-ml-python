#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import datetime
from typing import List, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model._deploy_client.utils import constants

_COMMON_PREFIX = "snowml_test_"


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
        if_not_exists: bool = False,
        or_replace: bool = False,
    ) -> str:
        actual_db_name = identifier.get_inferred_name(db_name)
        or_replace_sql = " OR REPLACE" if or_replace else ""
        if_not_exists_sql = " IF NOT EXISTS" if if_not_exists else ""
        self._session.sql(
            f"CREATE{or_replace_sql} DATABASE{if_not_exists_sql} {actual_db_name} DATA_RETENTION_TIME_IN_DAYS = 0"
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
        if_not_exists: bool = False,
        or_replace: bool = False,
    ) -> str:
        actual_schema_name = identifier.get_inferred_name(schema_name)
        if db_name:
            actual_db_name = self.create_database(db_name, if_not_exists=True)
            full_qual_schema_name = f"{actual_db_name}.{actual_schema_name}"
        else:
            full_qual_schema_name = actual_schema_name
        or_replace_sql = " OR REPLACE" if or_replace else ""
        if_not_exists_sql = " IF NOT EXISTS" if if_not_exists else ""
        self._session.sql(f"CREATE{or_replace_sql} SCHEMA{if_not_exists_sql} {full_qual_schema_name}").collect()
        return full_qual_schema_name

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

    def cleanup_schemas(self, db_name: Optional[str] = None, expire_days: int = 3) -> None:
        schemas_df = self.show_schemas(f"{_COMMON_PREFIX}%", db_name)
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
        if_not_exists: bool = False,
        or_replace: bool = False,
        sse_encrypted: bool = False,
    ) -> str:
        actual_stage_name = identifier.get_inferred_name(stage_name)
        if schema_name:
            full_qual_schema_name = self.create_schema(schema_name, db_name, if_not_exists=True)
            full_qual_stage_name = f"{full_qual_schema_name}.{actual_stage_name}"
        else:
            full_qual_stage_name = actual_stage_name
        or_replace_sql = " OR REPLACE" if or_replace else ""
        if_not_exists_sql = " IF NOT EXISTS" if if_not_exists else ""
        encryption_sql = " ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')" if sse_encrypted else ""
        self._session.sql(
            f"CREATE{or_replace_sql} STAGE{if_not_exists_sql} {full_qual_stage_name}{encryption_sql}"
        ).collect()
        return full_qual_stage_name

    def show_stages(
        self,
        stage_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> snowpark.DataFrame:
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

    def create_image_repo(self, repo_name: str) -> None:
        self._session.sql(f"CREATE OR REPLACE IMAGE REPOSITORY {repo_name}").collect()

    def drop_image_repo(self, repo_name: str) -> None:
        self._session.sql(f"DROP IMAGE REPOSITORY IF EXISTS {repo_name}").collect()

    def show_user_functions(
        self,
        function_name: str,
        schema_name: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> snowpark.DataFrame:
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
            func_argments = str(stale_func.arguments)
            func_def = func_argments.partition("RETURN")[0].strip()
            self.drop_function(function_def=func_def, schema_name=schema_name, db_name=db_name, if_exists=True)

    def get_snowservice_image_repo(
        self,
        repo: str,
        subdomain: str = constants.DEV_IMAGE_REGISTRY_SUBDOMAIN,
    ) -> str:
        conn = self._session._conn._conn
        org = conn.host.split(".")[1]
        account = conn.account
        db = conn._database
        schema = conn._schema
        return f"{org}-{account}.{subdomain}.{constants.PROD_IMAGE_REGISTRY_DOMAIN}/{db}/{schema}/{repo}".lower()


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

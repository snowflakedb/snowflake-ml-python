import importlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.feature_store import Entity, FeatureView  # type: ignore[attr-defined]
from snowflake.snowpark import DataFrame, Session, functions as F
from snowflake.snowpark.types import TimestampTimeZone, TimestampType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExampleHelper:
    def __init__(self, session: Session, database_name: str, dataset_schema: str) -> None:
        """A helper class to run Feature Store examples.

        Args:
            session: A Snowpark session object.
            database_name: Database where dataset and Feature Store lives.
            dataset_schema: Schema where destination dataset table lives.
        """
        self._session = session
        self._database_name = database_name
        self._dataset_schema = dataset_schema
        self._clear()

    def _clear(self) -> None:
        self._selected_example = None
        self._source_tables: list[str] = []
        self._source_dfs: list[DataFrame] = []
        self._excluded_columns: list[sql_identifier.SqlIdentifier] = []
        self._label_columns: list[sql_identifier.SqlIdentifier] = []
        self._timestamp_column: Optional[sql_identifier.SqlIdentifier] = None
        self._epoch_to_timestamp_cols: list[str] = []
        self._add_id_column: Optional[sql_identifier.SqlIdentifier] = None
        self._training_spine_table: str = ""

    def list_examples(self) -> Optional[DataFrame]:
        """Return a dataframe object about descriptions of all examples."""
        root_dir = Path(__file__).parent
        rows = []
        hide_folders = ["citibike_trip_features", "source_data"]
        for f_name in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, f_name)) and f_name[0].isalpha() and f_name not in hide_folders:
                source_file_path = root_dir.joinpath(f"{f_name}/source.yaml")
                source_dict = self._read_yaml(str(source_file_path))
                rows.append((f_name, source_dict["model_category"], source_dict["desc"], source_dict["label_columns"]))
        return self._session.create_dataframe(rows, schema=["NAME", "MODEL_CATEGORY", "DESC", "LABEL_COLS"])

    def load_draft_feature_views(self) -> list[FeatureView]:
        """Return all feature views in an example.

        Returns:
            A list of FeatureView object.
        """
        fvs = []
        root_dir = Path(__file__).parent.joinpath(f"{self._selected_example}/features")
        for f_name in os.listdir(root_dir):
            if not f_name[0].isalpha():
                # skip folders like __pycache__
                continue
            mod_path = f"{__package__}.{self._selected_example}.features.{f_name.rstrip('.py')}"
            mod = importlib.import_module(mod_path)
            fv = mod.create_draft_feature_view(
                self._session, self._source_dfs, self._source_tables, self._database_name, self._dataset_schema
            )
            fvs.append(fv)

        return fvs

    def load_entities(self) -> list[Entity]:
        """Return all entities in an example.

        Returns:
            A list of Entity object.
        """
        current_module = f"{__package__}.{self._selected_example}.entities"
        mod = importlib.import_module(current_module)
        return mod.get_all_entities()  # type: ignore[no-any-return]

    def _read_yaml(self, file_path: str) -> Any:
        with open(file_path) as fs:
            return yaml.safe_load(fs)

    def _create_file_format(self, format_dict: dict[str, str], format_name: str) -> None:
        """Create a file name with given name."""
        self._session.sql(
            f"""
            create or replace file format {format_name}
                type = '{format_dict['type']}'
                compression = '{format_dict['compression']}'
                field_delimiter = '{format_dict['field_delimiter']}'
                record_delimiter = '{format_dict['record_delimiter']}'
                skip_header = {format_dict['skip_header']}
                field_optionally_enclosed_by = '{format_dict['field_optionally_enclosed_by']}'
                trim_space = {format_dict['trim_space']}
                error_on_column_count_mismatch = {format_dict['error_on_column_count_mismatch']}
                escape = '{format_dict['escape']}'
                escape_unenclosed_field = '{format_dict['escape_unenclosed_field']}'
                date_format = '{format_dict['date_format']}'
                timestamp_format = '{format_dict['timestamp_format']}'
                null_if = {format_dict['null_if']}
                comment = '{format_dict['comment']}'
            """
        ).collect()

    def _load_csv(self, schema_dict: dict[str, str], temp_stage_name: str) -> list[str]:
        # create temp file format
        file_format_name = f"{self._database_name}.{self._dataset_schema}.feature_store_temp_format"
        format_str = ""
        if "format" in schema_dict:
            self._create_file_format(schema_dict["format"], file_format_name)  # type: ignore[arg-type]
            format_str = f"file_format = {file_format_name}"

        # create destination table
        cols_type_str = ",".join([f"{k} {v}" for k, v in schema_dict["columns"].items()])  # type: ignore[attr-defined]
        cols_name_str = ",".join(schema_dict["columns"].keys())  # type: ignore[attr-defined]
        if self._add_id_column:
            cols_type_str = (
                f"{self._add_id_column.resolved()} number autoincrement start 1 increment 1, " + cols_type_str
            )

        destination_table = f"{self._database_name}.{self._dataset_schema}.{schema_dict['destination_table_name']}"
        self._session.sql(
            f"""
            create or replace table {destination_table} ({cols_type_str})
            """
        ).collect()

        # copy dataset on stage into destination table
        self._session.sql(
            f"""
            copy into {destination_table} ({cols_name_str}) from
                @{temp_stage_name}
                {format_str}
                pattern = '{schema_dict['load_files_pattern']}'
            """
        ).collect()

        return [schema_dict["destination_table_name"]]

    def _load_parquet(self, schema_dict: dict[str, str], temp_stage_name: str) -> list[str]:
        regex_pattern = schema_dict["load_files_pattern"]
        all_files = self._session.sql(f"list @{temp_stage_name}").collect()
        filtered_files = [item["name"] for item in all_files if re.match(regex_pattern, item["name"])]
        file_count = len(filtered_files)
        result = []

        for file in filtered_files:
            file_name = file.rsplit("/", 1)[-1]

            df = self._session.read.parquet(f"@{temp_stage_name}/{file_name}")
            for old_col_name in df.columns:
                df = df.with_column_renamed(old_col_name, identifier.get_unescaped_names(old_col_name))

            # convert timestamp column to ntz
            for name, type in dict(df.dtypes).items():
                if type == "timestamp":
                    df = df.with_column(name, F.to_timestamp_ntz(name))

            # convert epoch column to ntz timestamp
            for ts_col in self._epoch_to_timestamp_cols:
                if "timestamp" != dict(df.dtypes)[ts_col]:
                    df = df.with_column(ts_col, F.cast(df[ts_col] / 1000000, TimestampType(TimestampTimeZone.NTZ)))

            if self._add_id_column:
                df = df.withColumn(self._add_id_column, F.monotonically_increasing_id())

            if file_count == 1:
                dest_table_name = (
                    f"{self._database_name}.{self._dataset_schema}.{schema_dict['destination_table_name']}"
                )
                result.append(schema_dict["destination_table_name"])
            else:
                regex_pattern = schema_dict["destination_table_name"]
                dest_table_name = re.match(regex_pattern, file_name).group("table_name")  # type: ignore[union-attr]
                result.append(dest_table_name)
                dest_table_name = f"{self._database_name}.{self._dataset_schema}.{dest_table_name}"

            df.write.mode("overwrite").save_as_table(dest_table_name)

        return result

    def _load_source_data(self, schema_yaml_file: str) -> list[str]:
        """Parse a yaml schema file and load data into Snowflake.

        Args:
            schema_yaml_file: the path to a yaml schema file.

        Returns:
            Return a destination table name.
        """
        # load schema file
        schema_dict = self._read_yaml(schema_yaml_file)
        temp_stage_name = f"{self._database_name}.{self._dataset_schema}.feature_store_temp_stage"

        # create a temp stage from S3 URL
        self._session.sql(f"create or replace stage {temp_stage_name} url = '{schema_dict['s3_url']}'").collect()

        # load csv or parquet
        # TODO: this could be more flexible and robust.
        if "parquet" in schema_dict["load_files_pattern"]:
            return self._load_parquet(schema_dict, temp_stage_name)
        else:
            return self._load_csv(schema_dict, temp_stage_name)

    def load_example(self, example_name: str) -> list[str]:
        """Select the active example and load its datasets to Snowflake.

        Args:
            example_name: The folder name under feature_store/examples.
                For example, 'citibike_trip_features'.

        Returns:
            Returns a list of table names with populated datasets.
        """
        self._clear()
        self._selected_example = example_name  # type: ignore[assignment]

        # load source yaml file
        root_dir = Path(__file__).parent
        source_file_path = root_dir.joinpath(f"{self._selected_example}/source.yaml")
        source_dict = self._read_yaml(str(source_file_path))
        self._source_tables = []
        self._source_dfs = []

        source_yaml_data = source_dict["source_data"]
        if "excluded_columns" in source_dict:
            self._excluded_columns = sql_identifier.to_sql_identifiers(source_dict["excluded_columns"].split(","))
        if "label_columns" in source_dict:
            self._label_columns = sql_identifier.to_sql_identifiers(source_dict["label_columns"].split(","))
        if "timestamp_column" in source_dict:
            self._timestamp_column = sql_identifier.SqlIdentifier(source_dict["timestamp_column"])
        if "epoch_to_timestamp_cols" in source_dict:
            self._epoch_to_timestamp_cols = source_dict["epoch_to_timestamp_cols"].split(",")
        if "add_id_column" in source_dict:
            self._add_id_column = sql_identifier.SqlIdentifier(source_dict["add_id_column"])
        self._training_spine_table = (
            f"{self._database_name}.{self._dataset_schema}.{source_dict['training_spine_table']}"
        )

        return self.load_source_data(source_yaml_data)

    def load_source_data(self, source_data_name: str) -> list[str]:
        """Load source data into Snowflake.

        Args:
            source_data_name: The name of source data located in examples/source_data/.

        Returns:
            Return a list of Snowflake tables.
        """
        root_dir = Path(__file__).parent
        schema_file = root_dir.joinpath(f"source_data/{source_data_name}.yaml")
        destination_tables = self._load_source_data(str(schema_file))
        for dest_table in destination_tables:
            source_df = self._session.table(dest_table)
            self._source_tables.append(dest_table)
            self._source_dfs.append(source_df)
            logger.info(f"{dest_table} has been created successfully.")
        return self._source_tables

    def get_current_schema(self) -> str:
        return self._dataset_schema

    def get_label_cols(self) -> list[str]:
        return [item.resolved() for item in self._label_columns]

    def get_excluded_cols(self) -> list[str]:
        return [item.resolved() for item in self._excluded_columns]

    def get_training_data_timestamp_col(self) -> Optional[str]:
        return self._timestamp_column.resolved() if self._timestamp_column is not None else None

    def get_training_spine_table(self) -> str:
        return self._training_spine_table

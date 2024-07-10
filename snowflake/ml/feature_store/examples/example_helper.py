import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from snowflake.snowpark import Session

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
        self._selected_example = None

    def list_examples(self) -> List[str]:
        """Return a list of examples."""
        root_dir = Path(__file__).parent
        result = []
        for f_name in os.listdir(root_dir):
            if os.path.isdir(f_name) and f_name[0].isalpha() and f_name != "source_data":
                result.append(f_name)
        return result

    def select_example(self, name: str) -> None:
        """Set given example as current active example.

        Args:
            name: The folder name under feature_store/examples.
                For example, 'recommender_system'.
        """
        self._selected_example = name

    def load_draft_feature_views(self) -> List[Any]:
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
            mod_path = f"{__package__}.{self._selected_example}.features.{f_name.strip('.py')}"
            mod = importlib.import_module(mod_path)
            fv = mod.create_draft_feature_view(self._session, f"{self._database_name}.{self._dataset_schema}")
            fvs.append(fv)

        return fvs

    def load_entities(self) -> List[Any]:
        """Return all entities in an example.

        Returns:
            A list of Entity object.
        """
        current_module = f"{__package__}.{self._selected_example}.entities"
        mod = importlib.import_module(current_module)
        return mod.get_all_entities()

    def _read_yaml(self, file_path: str) -> Dict[str, str]:
        with open(file_path) as fs:
            return yaml.safe_load(fs)

    def _create_file_format(self, format_dict, format_name) -> None:
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

    def _load_source_data(self, schema_yaml_file) -> str:
        """Parse a yaml schema file and load data into Snowflake.

        Args:
            schema_yaml_file: the path to a yaml schema file.

        Returns:
            Return a destination table name.
        """
        # load schema file
        schema_dict = self._read_yaml(schema_yaml_file)

        file_format_name = f"{self._database_name}.{self._dataset_schema}.feature_store_temp_format"
        temp_stage_name = f"{self._database_name}.{self._dataset_schema}.feature_store_temp_stage"
        destination_table = f"{self._database_name}.{self._dataset_schema}.{schema_dict['destination_table_name']}"

        # create temp file format
        self._create_file_format(schema_dict["format"], file_format_name)

        # create a temp stage from S3 URL
        self._session.sql(f"create or replace stage {temp_stage_name} url = '{schema_dict['s3_url']}'").collect()

        # create destination table
        columns_type_str = ",".join([f"{k} {v}" for k, v in schema_dict["columns"].items()])
        self._session.sql(
            f"""
            create or replace table {destination_table} ({columns_type_str})
            """
        ).collect()

        # copy dataset on stage into destination table
        self._session.sql(
            f"""
            copy into {destination_table} from
                @{temp_stage_name}
                file_format = {file_format_name}
                pattern = '{schema_dict['load_files_pattern']}'
            """
        ).collect()

        return destination_table

    def setup_datasets(self) -> List[str]:
        """Load datasets to Snowflake and save as tables.

        Returns:
            A list of table names populated with dataset in Snowflake.
        """
        root_dir = Path(__file__).parent

        # load source yaml file
        source_file_path = root_dir.joinpath(f"{self._selected_example}/source.yaml")
        source_dict = self._read_yaml(source_file_path)
        result_tables = []

        for yaml_file in source_dict.keys():
            schema_file = root_dir.joinpath(f"source_data/{yaml_file}.yaml")
            destination_table = self._load_source_data(schema_file)
            result_tables.append(destination_table)
            logger.info(f"source data {yaml_file} has been successfully loaded into table {destination_table}.")

        return result_tables

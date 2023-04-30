import argparse
import json
import time
from os import path
from typing import Optional

from absl import logging

import snowflake.connector
from snowflake.ml.utils import connection_params


class ResultUploader:
    """Connect to the database via credientials,
    Read from the json file, which is generated from parse_coverage.py,
    Insert into {db}.{schema}.{table}
    """

    def __init__(
        self,
        breakdown_coverage_file_path: str,
        *,
        connection_file: str = "",
        connection_name: str = "",
        elapsed_time: int = -1,
        git_revision_number: str = "unknown",
    ) -> None:
        """Initializer.

        Args:
            breakdown_coverage_file_path: The path to the breakdown coverage JSON file.
                The output from `parse_coverage` script
            connection_file: File that contains the connection information,
                such as database, username, password, etc.
            connection_name: The name of connection.
                See https://docs.snowflake.com/en/user-guide/snowsql-start.html#configuring-default-connection-settings.
            elapsed_time: Runtime of the bazel code coverage.
            git_revision_number: Git revision number for the current run.

        Raises:
            ValueError: when output_directory or connection_file doesn't exist
            ValueError: when output_directory or connection_file doesn't exist
        """
        self._breakdown_coverage_file_path = breakdown_coverage_file_path  # Output directory for json files

        if not path.exists(self._breakdown_coverage_file_path):
            raise ValueError(f"input file: {self._breakdown_coverage_file_path} not found! closing session...")
        logging.info("input file: %s", self._breakdown_coverage_file_path)

        self._connection_file = connection_file
        self._connection_name = connection_name

        self._elapsed_time = elapsed_time
        self._git_revision_number = git_revision_number
        self._cursor: Optional[snowflake.connector.cursor.SnowflakeCursor] = None

    def _get_config(self) -> None:
        # Read the config to connect
        self._conn_cfg = connection_params.SnowflakeLoginOptions(
            connection_name=self._connection_name, login_file=self._connection_file
        )

    def _connect(self) -> None:
        self._cursor = snowflake.connector.connect(**self._conn_cfg).cursor()
        logging.info("Connected to the database!")

    def _write_to_table(self, db: str, schema: str, table: str = "breakdown_coverage") -> None:
        """Write to table

        Args:
            db: database name, given in the config
            schema: schema name, given in the config
            table: table name.
        """ """"""
        logging.info("retrieving result files")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        temp_table = f"temp_{table}"

        assert self._cursor is not None

        # create temporary table to help de-duplicate the query inserting into table
        # de-duplicate target at : git revision string
        self._cursor.execute(
            f"""CREATE TEMPORARY TABLE {db}.{schema}.{temp_table}
            (
                time TIMESTAMP_LTZ,
                git_revision VARCHAR(40),
                package VARCHAR,
                filename VARCHAR,
                covered_lines INT,
                total_lines INT,
                duration_sec INT
            )
            """
        )

        with open(self._breakdown_coverage_file_path) as f:
            breakdown_coverage = json.load(f)
            breakdown_info = []
            for each_breakdown_info in breakdown_coverage:
                # check that these keys exist in the json dict
                for json_key in ("package", "filename", "line_hit", "line_found"):
                    if json_key not in each_breakdown_info.keys():
                        logging.error(f"Key: {json_key} not in breakdown_coverage.json!")
                breakdown_info.append(
                    (
                        current_time,
                        self._git_revision_number,
                        each_breakdown_info["package"],
                        each_breakdown_info["filename"],
                        each_breakdown_info["line_hit"],
                        each_breakdown_info["line_found"],
                        self._elapsed_time,
                    )
                )

            # Multiple insert would be slower because of the connection overhead.
            # Doing multiple inserts at once will reduce the cost of overhead per insert.
            breakdown_sql = f"""
                insert into {db}.{schema}.{temp_table}
                (time, git_revision, package, filename, covered_lines, total_lines, duration_sec)
                values
                (%s, %s, %s, %s, %s, %s, %s)
                """
            self._cursor.executemany(breakdown_sql, breakdown_info)
            logging.info("done writing to temp table")

        # Merge into the target table so that if there are duplicate entries, which can be caused by
        # running this program multiple times against the same git revision, they will not be inserted.
        merge_sql = f"""
            MERGE INTO {db}.{schema}.{table}
            USING (
                SELECT
                time, git_revision, package, filename, covered_lines, total_lines, duration_sec
                FROM {db}.{schema}.{temp_table}
                ) AS b
                ON {table} .git_revision = b.git_revision
            WHEN NOT MATCHED THEN
                INSERT (time, git_revision, package, filename, covered_lines, total_lines, duration_sec)
                VALUES (b.time, b.git_revision, b.package, b.filename, b.covered_lines, b.total_lines, b.duration_sec)
        """
        self._cursor.execute(merge_sql)
        if self._cursor.fetchall()[0][0] > 0:
            logging.info("done writing new main git revision coverage info to table!")
        else:
            logging.warning("main git revision string is the same, no new code coverage info to table!")

        self._cursor.execute(f"DROP TABLE IF EXISTS {db}.{schema}.{temp_table};")

    def upload(self) -> None:
        self._get_config()
        self._connect()
        try:
            self._write_to_table(self._conn_cfg["database"], self._conn_cfg["schema"])
        finally:
            assert self._cursor is not None
            self._cursor.close()
            logging.info("connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload code coverage info into database", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-b",
        "--breakdown-coverage-file",
        type=str,
        help="The path to the breakdown coverage JSON file.",
        required=True,
    )
    parser.add_argument("-c", "--connection-config", type=str, help="path to the connection config file", required=True)
    parser.add_argument("-n", "--connection-name", type=str, help="connection name", required=True)
    parser.add_argument("-e", "--elapsed-time", type=int, help="elapsed time for bazel coverage", required=True)
    parser.add_argument("-r", "--git-revision", type=str, help="git revision string for the current run", required=True)
    args = parser.parse_args()

    result_uploader = ResultUploader(
        args.breakdown_coverage_file,
        connection_file=args.connection_config,
        connection_name=args.connection_name,
        elapsed_time=args.elapsed_time,
        git_revision_number=args.git_revision,
    )
    result_uploader.upload()

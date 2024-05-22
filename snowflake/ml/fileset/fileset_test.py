import collections

from absl.testing import absltest

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml._internal.exceptions import fileset_errors
from snowflake.ml.fileset import fileset
from snowflake.ml.test_utils import mock_data_frame
from snowflake.snowpark import types

MockResultMetaData = collections.namedtuple("MockResultMetaData", ["name", "type_code", "precision", "scale"])


class FileSetTest(absltest.TestCase):
    def setUp(self) -> None:
        self.mock_connection = absltest.mock.MagicMock(spec=connection.SnowflakeConnection)

        # Manually add some missing artifacts to make sure the success of creating the snowpark session
        self.mock_connection.is_closed.return_value = False
        self.mock_connection._telemetry = absltest.mock.Mock()
        self.mock_connection._session_parameters = absltest.mock.Mock()

        self.df_collect_patcher = absltest.mock.patch(
            "snowflake.snowpark.dataframe.DataFrame.collect",
            side_effect=[mock_data_frame.MockAsyncJob(r) for r in ["random res", [], "random res"]],
        )
        self.mock_df = self.df_collect_patcher.start()

    def tearDown(self) -> None:
        self.df_collect_patcher.stop()

    def test_init_with_wrong_args(self) -> None:
        """Test if any error could be raised when FileSet is initiated with invalid args."""
        snowpark_session = snowpark.Session.builder.config("connection", self.mock_connection).create()
        input_kwargs_list = [
            {  # Absence of both the snowpark session and the snowflake python connection.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
            },
            {  # Presence of both the snowpark session and the snowflake python connection.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
                "snowpark_session": snowpark_session,
                "sf_connection": self.mock_connection,
            },
        ]
        for input_kwargs in input_kwargs_list:
            with self.assertRaises(ValueError):
                fileset.FileSet(**input_kwargs)

    def test_make_with_wrong_args(self) -> None:
        """Test if any error could be raised when FileSet.make() is called with invalid args."""
        snowpark_session = snowpark.Session.builder.config("connection", self.mock_connection).create()
        query = "SELECT whatever FROM Mytable"
        df = snowpark_session.sql(query)
        input_kwargs_list = [
            {  # Absence of both the snowpark dataframe and the snowflake python connection.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
            },
            {  # Presence of both the snowpark dataframe and the snowflake python connection.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
                "snowpark_dataframe": df,
                "sf_connection": self.mock_connection,
            },
            {  # Presence of both the snowpark dataframe and the snowflake python connection.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
                "snowpark_dataframe": df,
                "sf_connection": self.mock_connection,
                "query": query,
            },
            {  # Absence of the SQL query when a snowflake python connection is given.
                "target_stage_loc": "@mydb.mychema.mystage/",
                "name": "test",
                "sf_connection": self.mock_connection,
            },
        ]
        for input_kwargs in input_kwargs_list:
            with self.assertRaises(ValueError):
                fileset.FileSet.make(**input_kwargs)

    def test_make_and_cast(self) -> None:
        """Test if a FileSet can be created from a make call and supported data type can be casted correctly."""
        test_cases = [
            ("@mydb.mychema.mystage/mydir/", "test", "number_int", types.DecimalType(15, 0), "BIGINT"),
            ("@mydb.mychema.mystage/", "test_fs", "number_float", types.DecimalType(15, 9), "FLOAT"),
            ("@db.schema.stage/", "test_fs", "tiny", types.ByteType(), "SMALLINT"),
            ("@db.schema.stage/", "test_fs", "short", types.ShortType(), "SMALLINT"),
            ("@db.schema.stage/", "test_fs", "integer", types.IntegerType(), "INT"),
            ("@db.schema.stage/", "test_fs", "long", types.LongType(), "BIGINT"),
        ]

        for test_case in test_cases:
            with self.subTest():
                with absltest.mock.patch("snowflake.ml.fileset.sfcfs.SFFileSystem", autospec=True) as MockSFFileSystem:
                    self.mock_df.side_effect = ["random", [], "random"]
                    instance = MockSFFileSystem.return_value
                    stage_location, fileset_name, col_name, col_type, expected_type = test_case
                    mock_cursor = self.mock_connection.cursor.return_value
                    instance.ls.return_value = [
                        f"{stage_location}/{fileset_name}/data_01aa0162-0405-9f0d-000c_015_1_0.snappy.parquet",
                        f"{stage_location}/{fileset_name}/data_01aa0162-0405-9f0d-000c_015_1_1.snappy.parquet",
                    ]

                    # Use the mock snowflake connection to build a dataframe.
                    snowpark_session = snowpark.Session.builder.config("connection", self.mock_connection).create()
                    df = snowpark_session.sql(f"SELECT {col_name} FROM Mytable")

                    # Manually set the schema of the dataframe to enable dataframe plan analyzer.
                    mock_struct_fields = [
                        types.StructField(col_name, col_type, True),
                    ]
                    df.schema = types.StructType._from_attributes(mock_struct_fields)

                    # FileSet.make() will make the dataframe to generate the query plan and use
                    # the mock connection to execute the query
                    fileset.FileSet.make(target_stage_loc=stage_location, name=fileset_name, snowpark_dataframe=df)
                    expected_query = (
                        f" COPY  INTO '{stage_location}' FROM ("
                        f' SELECT  CAST ("{col_name.upper()}" AS {expected_type}) AS "{col_name.upper()}" FROM'
                        f" (SELECT {col_name} FROM Mytable))"
                        f" PARTITION BY '{fileset_name}'"
                        " FILE_FORMAT  = (  TYPE  = parquet )  max_file_size = 33554432 detailed_output = True "
                        "  HEADER  = True"
                    )
                    mock_cursor.execute.assert_called_with(
                        expected_query, params=absltest.mock.ANY, _statement_params=absltest.mock.ANY
                    )

                    # FileSet.make() will shuffle rows by random if shuffle is true
                    mock_cursor.execute.reset_mock()
                    self.mock_df.side_effect = ["random", [], "random"]
                    with absltest.mock.patch("snowflake.snowpark.functions.randint", return_value=1):
                        fileset.FileSet.make(
                            target_stage_loc=stage_location, name=fileset_name, snowpark_dataframe=df, shuffle=True
                        )
                        expected_query = (
                            f" COPY  INTO '{stage_location}' FROM ("
                            f' SELECT  CAST ("{col_name.upper()}" AS {expected_type}) AS "{col_name.upper()}" FROM'
                            f" (SELECT {col_name} FROM Mytable) ORDER BY random(1) ASC NULLS FIRST)"
                            f" PARTITION BY '{fileset_name}'"
                            " FILE_FORMAT  = (  TYPE  = parquet )  max_file_size = 33554432 detailed_output = True "
                            "  HEADER  = True"
                        )
                        mock_cursor.execute.assert_called_with(
                            expected_query, params=absltest.mock.ANY, _statement_params=absltest.mock.ANY
                        )

    def test_make_and_no_cast(self) -> None:
        """Test if a FileSet can be created from a make call and uncasted data type are not casted."""
        test_cases = [
            ("@db.schema.stage/", "test_fs", "float", types.FloatType()),
            ("@db.schema.stage/", "test_fs", "string", types.StringType()),
            ("@db.schema.stage/", "test_fs", "binary", types.BinaryType()),
            ("@db.schema.stage/", "test_fs", "bool", types.BooleanType()),
            ("@db.schema.stage/", "test_fs", "array", types.ArrayType()),
            ("@db.schema.stage/", "test_fs", "map", types.MapType()),
            ("@db.schema.stage/", "test_fs", "timestamp", types.TimestampType()),
            ("@db.schema.stage/", "test_fs", "time", types.TimeType()),
            ("@db.schema.stage/", "test_fs", "date", types.DateType()),
            ("@db.schema.stage/", "test_fs", "variant", types.VariantType()),
        ]

        for test_case in test_cases:
            with self.subTest():
                with absltest.mock.patch("snowflake.ml.fileset.sfcfs.SFFileSystem", autospec=True) as MockSFFileSystem:
                    self.mock_df.side_effect = ["random", [], "random"]
                    instance = MockSFFileSystem.return_value
                    stage_location, fileset_name, col_name, col_type = test_case
                    mock_cursor = self.mock_connection.cursor.return_value
                    instance.ls.return_value = [
                        f"{stage_location}/{fileset_name}/data_01aa0162-0405-9f0d-000c_015_1_0.snappy.parquet",
                        f"{stage_location}/{fileset_name}/data_01aa0162-0405-9f0d-000c_015_1_1.snappy.parquet",
                    ]

                    # Use the mock snowflake connection to build a dataframe.
                    snowpark_session = snowpark.Session.builder.config("connection", self.mock_connection).create()
                    df = snowpark_session.sql(f"SELECT {col_name} FROM Mytable")

                    # Manually set the schema of the dataframe to enable dataframe plan analyzer.
                    mock_struct_fields = [
                        types.StructField(col_name, col_type, True),
                    ]
                    df.schema = types.StructType._from_attributes(mock_struct_fields)

                    # FileSet.make() will make the dataframe to generate the query plan and use
                    # the mock connection to execute the query
                    fileset.FileSet.make(target_stage_loc=stage_location, name=fileset_name, snowpark_dataframe=df)
                    expected_query = (
                        f" COPY  INTO '{stage_location}' FROM ( SELECT \"{col_name.upper()}\""
                        f" FROM (SELECT {col_name} FROM Mytable))"
                        f" PARTITION BY '{fileset_name}'"
                        " FILE_FORMAT  = (  TYPE  = parquet )  max_file_size = 33554432 detailed_output = True "
                        "  HEADER  = True"
                    )
                    mock_cursor.execute.assert_called_with(
                        expected_query, params=absltest.mock.ANY, _statement_params=absltest.mock.ANY
                    )

    def test_make_fail_by_files_from_different_query(self) -> None:
        """Test if a FileSet creation could fail because files contain different query id."""
        with absltest.mock.patch("snowflake.ml.fileset.sfcfs.SFFileSystem", autospec=True) as MockSFFileSystem:
            with self.assertRaises(fileset_errors.MoreThanOneQuerySourceError):
                instance = MockSFFileSystem.return_value
                instance.ls.return_value = [
                    "@mydb.mychema.mystage/mydir/test/data_01aa0162-0405-9f0d-000c-a90103dfc8_015_1_0.snappy.parquet",
                    "@mydb.mychema.mystage/mydir/test/data_02aa0162-0405-9f0d-000c-a90103dfc8_015_1_1.snappy.parquet",
                ]
                fileset.FileSet(
                    target_stage_loc="@mydb.mychema.mystage/mydir",
                    name="test",
                    sf_connection=self.mock_connection,
                )

    def test_make_fail_by_cse_stage(self) -> None:
        """Test if a FileSet creation could fail because the target stage is not a server side encrypted stage."""
        self.mock_df.side_effect = [[]]
        with self.assertRaises(fileset_errors.FileSetLocationError):
            fileset.FileSet(
                target_stage_loc="@mydb.mychema.mystage/mydir",
                name="test",
                sf_connection=self.mock_connection,
            )

    def test_files(self) -> None:
        """Test if a FileSet can return a list of stage file paths in sfc protocol."""
        with absltest.mock.patch("snowflake.ml.fileset.sfcfs.SFFileSystem", autospec=True) as MockSFFileSystem:
            instance = MockSFFileSystem.return_value
            instance.ls.return_value = [
                "@mydb.mychema.mystage/mydir/test/data_01aa0162-0405-9f0d-000c-a90103dfc8_015_1_0.snappy.parquet",
                "@mydb.mychema.mystage/mydir/test/data_01aa0162-0405-9f0d-000c-a90103dfc8_015_1_1.snappy.parquet",
            ]
            test_fileset = fileset.FileSet(
                target_stage_loc="@mydb.mychema.mystage/mydir",
                name="test",
                sf_connection=self.mock_connection,
            )
            expected_files = [
                "sfc://@mydb.mychema.mystage/mydir/test/data_01aa0162-0405-9f0d-000c-a90103dfc8_015_1_0.snappy.parquet",
                "sfc://@mydb.mychema.mystage/mydir/test/data_01aa0162-0405-9f0d-000c-a90103dfc8_015_1_1.snappy.parquet",
            ]
            self.assertListEqual(expected_files, test_fileset.files())

    def test_fileset_stage_location(self) -> None:
        """Test if a FileSet can return its location in sfc protocol."""
        test_fileset = fileset.FileSet(
            target_stage_loc="@mydb.mychema.mystage/mydir",
            name="test",
            sf_connection=self.mock_connection,
        )
        expected_loc = "sfc://@mydb.mychema.mystage/mydir/test/"
        self.assertEqual(expected_loc, test_fileset.fileset_stage_location())

    def test_delete(self) -> None:
        """Test if a FileSet stage can be deleted by calling delete()."""

        test_fileset = fileset.FileSet(
            target_stage_loc="@mydb.mychema.mystage/mydir",
            name="test",
            sf_connection=self.mock_connection,
        )
        test_fileset.delete()
        self.assertEmpty(test_fileset._files)
        self.assertRaises(fileset_errors.FileSetAlreadyDeletedError, test_fileset.delete)
        self.assertRaises(fileset_errors.FileSetAlreadyDeletedError, test_fileset.to_torch_datapipe, shuffle=True)
        self.assertRaises(fileset_errors.FileSetAlreadyDeletedError, test_fileset.files)
        self.assertRaises(fileset_errors.FileSetAlreadyDeletedError, test_fileset.fileset_stage_location)
        self.assertRaises(fileset_errors.FileSetAlreadyDeletedError, test_fileset.to_tf_dataset)

    def test_invalid_stage_path(self) -> None:
        """Test if an error can be raised if the FileSet is given an invalid stage path."""
        with self.assertRaises(fileset_errors.FileSetLocationError):
            # The stage path should start with "@"
            fileset.FileSet(
                target_stage_loc="mydb.mychema.mystage/mydir",
                name="test",
                sf_connection=self.mock_connection,
            )
        with self.assertRaises(fileset_errors.FileSetLocationError):
            # The stage path should be in the form "@<database>.<schema>.<stage>/*"
            fileset.FileSet(
                target_stage_loc="@mystage/mydir",
                name="test",
                sf_connection=self.mock_connection,
            )


if __name__ == "__main__":
    absltest.main()

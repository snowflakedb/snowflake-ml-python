import fsspec
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml.fileset import snowfs


class SnowFileSystemTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.mock_connection = absltest.mock.MagicMock(spec=connection.SnowflakeConnection)
        self.mock_connection._telemetry = absltest.mock.Mock()
        self.mock_connection._session_parameters = absltest.mock.Mock()

        # Manually add some missing artifacts to make sure the success of creating the snowpark session
        self.mock_connection.is_closed.return_value = False
        self.snowpark_session = snowpark.Session.builder.config("connection", self.mock_connection).create()

    def test_init_sf_file_system(self) -> None:
        """Test if the FS could be initialized with a snowpark session or a snowflake python connection."""

        sffs1 = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)
        sffs2 = snowfs.SnowFileSystem(sf_connection=self.mock_connection)
        self.assertEqual(sffs1._conn, sffs2._conn)

        with self.assertRaises(ValueError):
            snowfs.SnowFileSystem()

    def test_optimize_read(self) -> None:
        """Test if optimize_read() can call correct stage filesystems to do their read optimization."""
        with absltest.mock.patch(
            "snowflake.ml.fileset.embedded_stage_fs.SFEmbeddedStageFileSystem", autospec=True
        ) as MockSFEmbeddedStageFileSystem:
            instance1 = absltest.mock.MagicMock()
            instance2 = absltest.mock.MagicMock()
            MockSFEmbeddedStageFileSystem.side_effect = [instance1, instance2]
            sffs = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)

            # Confirm optimize_read() will do nothing for empty input
            sffs.optimize_read([])
            MockSFEmbeddedStageFileSystem.assert_not_called()

            file_list = [
                "snow://dataset/testdb.testschema.foo/versions/nytrain/a",
                "snow://dataset/testdb.testschema.foo/versions/nytrain/b",
                "snow://dataset/testdb.testschema.bar/versions/nytrain/c",
            ]
            sffs.optimize_read(file_list)
            MockSFEmbeddedStageFileSystem.assert_any_call(
                snowpark_session=self.snowpark_session,
                domain="dataset",
                name="testdb.testschema.bar",
            )
            MockSFEmbeddedStageFileSystem.assert_any_call(
                snowpark_session=self.snowpark_session,
                domain="dataset",
                name="testdb.testschema.foo",
            )
            instance1.optimize_read.assert_any_call(["versions/nytrain/a", "versions/nytrain/b"])
            instance2.optimize_read.assert_any_call(["versions/nytrain/c"])

    def test_open(self) -> None:
        """Test if 'open' is able to parse the input and call the underlying file system to open files."""
        with absltest.mock.patch(
            "snowflake.ml.fileset.embedded_stage_fs.SFEmbeddedStageFileSystem", autospec=True
        ) as MockSFEmbeddedStageFileSystem:
            # Test use case of initializing file system object and using it to read
            instance = MockSFEmbeddedStageFileSystem.return_value
            instance._open.return_value = absltest.mock.MagicMock(spec=fsspec.spec.AbstractBufferedFile)
            sffs = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)
            sffs.open("snow://dataset/testdb.testschema.foo/versions/nytrain/1.txt")
            MockSFEmbeddedStageFileSystem.assert_any_call(
                snowpark_session=self.snowpark_session,
                domain="dataset",
                name="testdb.testschema.foo",
            )
            instance._open.assert_called()
            instance._open.assert_any_call(
                "versions/nytrain/1.txt", mode="rb", block_size=None, autocommit=True, cache_options=None
            )

        with absltest.mock.patch(
            "snowflake.ml.fileset.embedded_stage_fs.SFEmbeddedStageFileSystem", autospec=True
        ) as MockSFEmbeddedStageFileSystem:
            # Test use case of reading from fsspec
            instance = MockSFEmbeddedStageFileSystem.return_value
            instance._open.return_value = absltest.mock.MagicMock(spec=fsspec.spec.AbstractBufferedFile)
            fp = fsspec.open(
                "snow://dataset/testdb.testschema.bar/versions/nytrain/1.txt", snowpark_session=self.snowpark_session
            )
            fp.open()
            MockSFEmbeddedStageFileSystem.assert_any_call(
                snowpark_session=self.snowpark_session,
                domain="dataset",
                name="testdb.testschema.bar",
            )
            instance._open.assert_called()
            instance._open.assert_any_call(
                "versions/nytrain/1.txt", mode="rb", block_size=None, autocommit=True, cache_options=None
            )

    @parameterized.parameters(  # type: ignore[misc]
        ("snow://dataset/foo/versions/1", "dataset", "foo", "1", ""),
        ("snow://test_domain/_testdb.testschema._foo/versions/1", "test_domain", "_testdb.testschema._foo", "1", ""),
        (
            'snow://_domain/testdb$."test""s""chema"._foo/versions/test_version',
            "_domain",
            'testdb$."test""s""chema"._foo',
            "test_version",
            "",
        ),
        (
            "snow://dataset/test1db.test$schema.foo/versions/nytrain/test/",
            "dataset",
            "test1db.test$schema.foo",
            "nytrain",
            "test/",
        ),
        (
            "snow://dataset/test_db.test_schema.foo/versions/nytrain/1.txt",
            "dataset",
            "test_db.test_schema.foo",
            "nytrain",
            "1.txt",
        ),
        ("dataset/foo/versions/_myVersion/some/nested/path", "dataset", "foo", "_myVersion", "some/nested/path"),
        (
            'test_domain/test_d$b."test.schema".foo$_o/versions/nytrain/subdir/file.ext',
            "test_domain",
            'test_d$b."test.schema".foo$_o',
            "nytrain",
            "subdir/file.ext",
        ),
        (
            'domain/"идентификатор"."test schema"."f.o_o1"/versions/_1/nytrain/',
            "domain",
            '"идентификатор"."test schema"."f.o_o1"',
            "_1",
            "nytrain/",
        ),
    )
    def test_parse_file_path(
        self, test_input: str, expected_domain: str, expected_name: str, expected_version: str, expected_relpath: str
    ) -> None:
        """Test if the FS could parse the snow URL location correctly"""
        expected_path = "/".join(s for s in ["versions", expected_version, expected_relpath] if s)
        with absltest.mock.patch(
            "snowflake.ml.fileset.embedded_stage_fs.SFEmbeddedStageFileSystem", autospec=True
        ) as MockSFEmbeddedStageFileSystem:
            instance = MockSFEmbeddedStageFileSystem.return_value
            sffs = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)
            sffs.ls(test_input)
            MockSFEmbeddedStageFileSystem.assert_any_call(
                snowpark_session=self.snowpark_session,
                domain=expected_domain,
                name=expected_name,
            )
            instance.ls.assert_any_call(expected_path, detail=True)

    @parameterized.parameters(  # type: ignore[misc]
        "snow://domain/versions/v1",  # Missing entity name
        "snow://test_db.test_schema.foo/versions/v1",  # Missing domain
        "snow://dataset/test_db.test_schema.foo/v1",  # Missing versions separator
        "snow://@test_db.test_schema.foo/v1",  # Invalid combination of snow URL with stage path
        "snow://sfc://@test_db.test_schema.foo/v1",  # Invalid combination of snow URL with stage path
        "sfc://snow://domain/test_db.test_schema.foo/versions/v1",  # Invalid combination of snow:// and sfc://
        'dataset/db."s"chema".foo/versions/v1',  # Double quoted identifier contains a single \"
        "dataset/3db.schema.foo/versions/v1/1.txt",  # Database name starts with digit
    )
    def test_negative_parse_file_path(self, test_input: str) -> None:
        """Test if the FS could fail the invalid input stage path"""
        with absltest.mock.patch("snowflake.ml.fileset.embedded_stage_fs.SFEmbeddedStageFileSystem", autospec=True):
            sffs = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)
            self.assertRaises(ValueError, sffs.ls, test_input)


if __name__ == "__main__":
    absltest.main()

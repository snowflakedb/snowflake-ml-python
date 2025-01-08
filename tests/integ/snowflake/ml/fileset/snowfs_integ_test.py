from uuid import uuid4

import fsspec
from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.exceptions import fileset_errors
from snowflake.ml.fileset import snowfs
from snowflake.ml.utils import connection_params


class TestSnowFileSystem(absltest.TestCase):
    """Integration tests for SnowFileSystem."""

    connection_parameters = connection_params.SnowflakeLoginOptions()
    snowpark_session = snowpark.Session.builder.configs(connection_parameters).create()
    sf_connection = snowpark_session._conn._conn
    db = snowpark_session.get_current_database()
    schema = snowpark_session.get_current_schema()

    domain = "dataset"
    entity = f"{db}.{schema}.snowfs_fbe_integ_{uuid4().hex}"
    version1 = "version1"
    version2 = "version2"
    row_counts = {
        version1: 10,
        version2: 20,
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.snowpark_session.sql(f"CREATE OR REPLACE {cls.domain} {cls.entity}").collect()
        _create_file_based_entity(
            cls.snowpark_session, "dataset", cls.entity, cls.version1, row_count=cls.row_counts[cls.version1], seed=42
        )
        _create_file_based_entity(
            cls.snowpark_session, "dataset", cls.entity, cls.version2, row_count=cls.row_counts[cls.version2], seed=43
        )

        cls.files = [
            f"{cls.domain}/{cls.entity}/{r['name']}"
            for version in [cls.version1, cls.version2]
            for r in cls.snowpark_session.sql(f"LIST 'snow://{cls.domain}/{cls.entity}/versions/{version}'").collect()
        ]
        assert len(cls.files) > 0, "LIST returned no files"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.snowpark_session.sql(f"DROP DATASET IF EXISTS {cls.entity}").collect()

    def setUp(self) -> None:
        # Create fs object with snowflake python connection
        self.sffs1 = snowfs.SnowFileSystem(sf_connection=self.sf_connection)
        # Create fs object with snowpark session
        self.sffs2 = snowfs.SnowFileSystem(snowpark_session=self.snowpark_session)

    def test_fs_creation(self) -> None:
        """Test if an object of Snowflake FS can be created."""
        # Create fs object with snowflake python connection via fsspec interface
        sffs3 = fsspec.filesystem("snow", sf_connection=self.sf_connection)
        # Create fs object with snowpark session via fsspec interface
        sffs4 = fsspec.filesystem("snow", snowpark_session=self.snowpark_session)

        # Due to instance caching, the object created directly should be the same object created via fsspec
        self.assertEqual(id(self.sffs1), id(sffs3))
        self.assertEqual(id(self.sffs2), id(sffs4))

    def test_fs_ls(self) -> None:
        """Test if ls() can list files in Snowflake stages."""
        for version in [self.version1, self.version2]:
            for path, expected in [
                (f"snow://{self.domain}/{self.entity}/versions/{version}/random", []),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}",
                    [
                        f"{self.domain}/{self.entity}/versions/{version}/test/",
                        f"{self.domain}/{self.entity}/versions/{version}/train/",
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/",
                    [
                        f"{self.domain}/{self.entity}/versions/{version}/test/",
                        f"{self.domain}/{self.entity}/versions/{version}/train/",
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/train",
                    [f for f in self.files if f.startswith(f"{self.domain}/{self.entity}/versions/{version}/train")],
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    self.assertListEqual(fs.ls(path), expected)

    def test_fs_ls_detail(self) -> None:
        """Test if ls(detail=True) can show file propertis in Snowflake stages."""
        for version in [self.version1, self.version2]:
            for path, expected in [
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}",
                    [
                        {
                            "name": f"{self.domain}/{self.entity}/versions/{version}/test/",
                            "type": "directory",
                            "size": 0,
                        },
                        {
                            "name": f"{self.domain}/{self.entity}/versions/{version}/train/",
                            "type": "directory",
                            "size": 0,
                        },
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/",
                    [
                        {
                            "name": f"{self.domain}/{self.entity}/versions/{version}/test/",
                            "type": "directory",
                            "size": 0,
                        },
                        {
                            "name": f"{self.domain}/{self.entity}/versions/{version}/train/",
                            "type": "directory",
                            "size": 0,
                        },
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/train",
                    [
                        {
                            "name": [
                                f
                                for f in self.files
                                if f.startswith(f"{self.domain}/{self.entity}/versions/{version}/train")
                            ][0],
                            "type": "file",
                        },
                    ],
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.ls(path, detail=True)
                    self.assertEqual(len(actual), len(expected))
                    for i in range(len(actual)):
                        actual_res = actual[i]
                        expected_res = expected[i]
                        if isinstance(actual_res, str):
                            raise AssertionError("ls() with detail should not return a list of strings.")
                        else:
                            self.assertDictContainsSubset(expected_res, actual_res)

    def test_fs_ls_on_trivial_stage(self) -> None:
        """Test if ls() can raise exception when the input points to invalid stages."""
        for fs in [self.sffs1, self.sffs2]:
            with self.assertRaises(fileset_errors.StageNotFoundError):
                fs.ls(f"{self.domain}/{self.entity}_not_exist/versions/")

    def test_open_from_snowfs_instance(self) -> None:
        """Test if stage files can be opened by SnowFileSystem instance."""
        for file in self.files:
            for fs in [self.sffs1, self.sffs2]:
                with fs.open(file, mode="rb") as f:
                    actual = f.read()
                    self.assertGreater(len(actual), 0)

    def test_open_from_fsspec(self) -> None:
        """Test if stage files can be opened through fsspec interface."""
        for file in self.files:
            with fsspec.open(f"snow://{file}", mode="rb", sf_connection=self.sf_connection) as f:
                actual = f.read()
                self.assertGreater(len(actual), 0)

            with fsspec.open(f"snow://{file}", mode="rb", snowpark_session=self.snowpark_session) as f:
                actual = f.read()
                self.assertGreater(len(actual), 0)

    def test_open_not_existing_file(self) -> None:
        """Test if open() can raise error when the given file cannot be found."""
        for fs in [self.sffs1, self.sffs2]:
            # Currently throwing StageNotFound error instead of matching sfcfs with StageFileNotFound
            with self.assertRaises(fileset_errors.StageNotFoundError):
                with fs.open(f"{self.domain}/{self.entity}_not_exist/versions/random/file_not_found") as f:
                    f.read()

    def test_info(self) -> None:
        """Test if snowfs can call info() to show properties of files."""
        for version in [self.version1, self.version2]:
            version_file = next(f for f in self.files if version in f)
            for path, expected in [
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/train",
                    {
                        "name": f"{self.domain}/{self.entity}/versions/{version}/train/",
                        "type": "directory",
                        "size": 0,
                    },
                ),
                (
                    version_file,
                    {
                        "name": version_file,
                        "type": "file",
                    },
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.info(path)
                    self.assertDictContainsSubset(expected, actual)

    def test_isdir(self) -> None:
        """Test if snowfs can call isdir() to check if a path is a directory."""
        for version in [self.version1, self.version2]:
            version_files = [f for f in self.files if version in f]
            for path, expected in [
                (f"snow://{self.domain}/{self.entity}/versions/{version}/train", True),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/test/", True),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/random", False),
            ] + [(f"snow://{f}", False) for f in version_files]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.isdir(path)
                    self.assertEqual(actual, expected)

    def test_isfile(self) -> None:
        """Test if snowfs can call isfile() to check if a path is a file."""
        for version in [self.version1, self.version2]:
            version_files = [f for f in self.files if version in f]
            for path, expected in [
                (f"snow://{self.domain}/{self.entity}/versions/{version}/train", False),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/test/", False),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/random", False),
            ] + [(f"snow://{f}", True) for f in version_files]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.isfile(path)
                    self.assertEqual(actual, expected)

    def test_exists(self) -> None:
        """Test if snowfs can call exists() to check if a path exists as a file or directory."""
        for version in [self.version1, self.version2]:
            version_files = [f for f in self.files if version in f]
            for path, expected in [
                (f"snow://{self.domain}/{self.entity}/versions/{version}/train", True),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/test/", True),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/random", False),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/test/random", False),
                (f"snow://{self.domain}/{self.entity}/versions/{version}/test/random/", False),
            ] + [(f"snow://{f}", True) for f in version_files]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.exists(path)
                    self.assertEqual(actual, expected)

    def test_find(self) -> None:
        """Test if snowfs can call find() to return all the files under a path."""
        for version in [self.version1, self.version2]:
            for path, expected in [
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/train/",
                    [
                        f"versions/{version}/train/data.*[.]parquet",
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/",
                    [
                        f"versions/{version}/test/data.*[.]parquet",
                        f"versions/{version}/train/data.*[.]parquet",
                    ],
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.find(path)
                    self.assertEqual(len(actual), len(expected))

    def test_du(self) -> None:
        """Test if snowfs can call du() to return the total size of files under a path."""
        for version in [self.version1, self.version2]:
            for fs in [self.sffs1, self.sffs2]:
                actual1 = fs.du(f"snow://{self.domain}/{self.entity}/versions/{version}")
                actual2 = fs.du(f"snow://{self.domain}/{self.entity}/versions/{version}/train")
                self.assertGreater(actual2, 0)
                self.assertGreater(actual1, actual2)

    def test_glob(self) -> None:
        """Test if sfsfs can call glob() to do wildcard matching."""
        for version in [self.version1, self.version2]:
            for pattern, expected in [
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/",
                    [
                        f"{self.domain}/{self.entity}/versions/{version}",
                    ],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/*",
                    [],  # Single asterisk only matches files, in this case nothing
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/**",
                    sorted(
                        [
                            f"{self.domain}/{self.entity}/versions/{version}",
                            f"{self.domain}/{self.entity}/versions/{version}/train/",
                            f"{self.domain}/{self.entity}/versions/{version}/test/",
                        ]
                        + [f for f in self.files if f.startswith(f"{self.domain}/{self.entity}/versions/{version}/")]
                    ),
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/**/*.parquet",
                    [f for f in self.files if f.startswith(f"{self.domain}/{self.entity}/versions/{version}/")],
                ),
                (
                    f"snow://{self.domain}/{self.entity}/versions/{version}/train/?ata*",
                    [f for f in self.files if f.startswith(f"{self.domain}/{self.entity}/versions/{version}/train")],
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    with self.subTest(f"pattern={pattern}"):
                        actual = fs.glob(pattern)
                        self.assertEqual(actual, expected)

    def test_negative_optimize_read(self) -> None:
        """Test if optimize_read() can raise error is presigned url fetching failed."""
        for fs in [self.sffs1, self.sffs2]:
            with self.assertRaises(fileset_errors.StageNotFoundError):
                fs.optimize_read([f"snow://{self.domain}/{self.entity}_not_exist/versions/foo/file.parquet"])


def _create_file_based_entity(
    session: snowpark.Session, domain: str, entity: str, version: str, row_count: int = 10, seed: int = 0
) -> None:
    session.sql(f"CREATE {domain} IF NOT EXISTS {entity}").collect()
    session.sql(
        f"ALTER {domain} {entity} ADD VERSION '{version}' FROM "
        f"(SELECT seq4() AS ID, uniform(1, 10, random({seed})) AS PART FROM TABLE(GENERATOR(ROWCOUNT => {row_count}))) "
        "PARTITION BY IFF(PART > 8, 'train', 'test')"
    ).collect()


if __name__ == "__main__":
    absltest.main()

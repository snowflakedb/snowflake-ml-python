import pickle

import fsspec
from absl.testing import absltest

from snowflake.ml._internal.exceptions import fileset_errors
from snowflake.ml.fileset import sfcfs
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.fileset import fileset_integ_utils


class TestSnowflakeFileSystem(absltest.TestCase):
    """Integration tests for Snowflake File System."""

    connection_parameters = connection_params.SnowflakeLoginOptions()
    snowpark_session = Session.builder.configs(connection_parameters).create()
    sf_connection = snowpark_session._conn._conn
    db = snowpark_session.get_current_database()
    schema = snowpark_session.get_current_schema()
    stage1 = f"{db}.{schema}.fileset_integ"
    stage2 = f"{db}.{schema}.fileset_integ_2"
    file_content1 = b"helloworld"
    file_content2 = b"helloworld again"

    @classmethod
    def setUpClass(cls) -> None:
        fileset_integ_utils.create_snowflake_stage_if_not_exists(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage1
        )
        fileset_integ_utils.upload_files_to_snowflake(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage1, content=cls.file_content1
        )
        fileset_integ_utils.create_snowflake_stage_if_not_exists(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage2
        )
        fileset_integ_utils.upload_files_to_snowflake(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage2, content=cls.file_content2
        )

    @classmethod
    def tearDownClass(cls) -> None:
        fileset_integ_utils.delete_files_from_snowflake_stage(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage1
        )
        fileset_integ_utils.delete_files_from_snowflake_stage(
            sp_session=cls.snowpark_session, snowflake_stage=cls.stage2
        )

    def setUp(self) -> None:
        # Create fs object with snowflake python connection
        self.sffs1 = sfcfs.SFFileSystem(sf_connection=self.sf_connection)
        # Create fs object with snowpark session
        self.sffs2 = sfcfs.SFFileSystem(snowpark_session=self.snowpark_session)

    def _get_content(self, stage: str) -> bytes:
        if stage == self.stage1:
            return self.file_content1
        elif stage == self.stage2:
            return self.file_content2
        return b""

    def test_fs_creation(self) -> None:
        """Test if an object of Snowflake FS can be created."""
        # Create fs object with snowflake python connection via fssepc interface
        sffs3 = fsspec.filesystem("sfc", sf_connection=self.sf_connection)
        # Create fs object with snowpark session via fssepc interface
        sffs4 = fsspec.filesystem("sfc", snowpark_session=self.snowpark_session)

        # Due to instance caching, the object created directly should be the same object createrd via fsspec
        self.assertEqual(id(self.sffs1), id(sffs3))
        self.assertEqual(id(self.sffs2), id(sffs4))

    def test_fs_ls(self) -> None:
        """Test if ls() can list files in Snowflake stages."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (f"@{stage}/random", []),
                (f"@{stage}", [f"@{stage}/test/", f"@{stage}/train/"]),
                (f"@{stage}/", [f"@{stage}/test/", f"@{stage}/train/"]),
                (f"@{stage}/train", [f"@{stage}/train/dataset/", f"@{stage}/train/helloworld.txt"]),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    self.assertEqual(fs.ls(path), expected)

    def test_fs_ls_detail(self) -> None:
        """Test if ls(detail=True) can show file propertis in Snowflake stages."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (
                    f"@{stage}",
                    [
                        {"name": f"@{stage}/test/", "type": "directory", "size": 0},
                        {"name": f"@{stage}/train/", "type": "directory", "size": 0},
                    ],
                ),
                (
                    f"@{stage}/",
                    [
                        {"name": f"@{stage}/test/", "type": "directory", "size": 0},
                        {"name": f"@{stage}/train/", "type": "directory", "size": 0},
                    ],
                ),
                (
                    f"@{stage}/train",
                    [
                        {"name": f"@{stage}/train/dataset/", "type": "directory", "size": 0},
                        {
                            "name": f"@{stage}/train/helloworld.txt",
                            "type": "file",
                            "size": len(self._get_content(stage)),
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
                            self.assertGreaterEqual(actual_res.items(), expected_res.items())

    def test_fs_ls_on_trivial_stage(self) -> None:
        """Test if ls() can raise exception when the input points to invalid stages."""
        for fs in [self.sffs1, self.sffs2]:
            with self.assertRaises(fileset_errors.StageNotFoundError):
                fs.ls("@ML_DATASETS.public.stage_does_not_exist")

    def test_open_from_sfcfs_instance(self) -> None:
        """Test if stage files can be opened by SFFileSystem instance."""
        for stage in [self.stage1, self.stage2]:
            for file in [
                f"@{stage}/train/dataset/helloworld1.txt",
                f"@{stage}/train/dataset/helloworld2.txt",
                f"@{stage}/train/helloworld.txt",
                f"@{stage}/test/testhelloworld.txt",
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    with fs.open(file, mode="rb") as f:
                        actual = f.read()
                        self.assertEqual(actual, self._get_content(stage))

    def test_open_from_fsspec(self) -> None:
        """Test if stage files can be opened through fsspec interface."""
        for stage in [self.stage1, self.stage2]:
            for file in [
                f"@{stage}/train/dataset/helloworld1.txt",
                f"@{stage}/train/dataset/helloworld2.txt",
                f"@{stage}/train/helloworld.txt",
                f"@{stage}/test/testhelloworld.txt",
            ]:
                with fsspec.open(f"sfc://{file}", mode="rb", sf_connection=self.sf_connection) as f:
                    actual = f.read()
                    self.assertEqual(actual, self._get_content(stage))

                with fsspec.open(f"sfc://{file}", mode="rb", snowpark_session=self.snowpark_session) as f:
                    actual = f.read()
                    self.assertEqual(actual, self._get_content(stage))

    def test_open_not_existing_file(self) -> None:
        """Test if open() can raise error when the given file cannot be found."""
        for fs in [self.sffs1, self.sffs2]:
            with self.assertRaises(fileset_errors.StageFileNotFoundError):
                with fs.open(f"@{self.stage1}/random_file_not_found") as f:
                    f.read()

    def test_info(self) -> None:
        """Test if sfcfs can call info() to show properties of files."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (f"@{stage}/train/dataset", {"name": f"@{stage}/train/dataset/", "type": "directory", "size": 0}),
                (
                    f"@{stage}/train/helloworld.txt",
                    {"name": f"@{stage}/train/helloworld.txt", "type": "file", "size": len(self._get_content(stage))},
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.info(path)
                    self.assertGreaterEqual(actual.items(), expected.items())

    def test_isdir(self) -> None:
        """Test if sfcfs can call isdir() to check if a path is a directory."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (f"@{stage}/train/dataset/", True),
                (f"@{stage}/train/dataset", True),
                (f"@{stage}/train", True),
                (f"@{stage}/test/", True),
                (f"@{stage}/train/dataset/helloworld1.txt", False),
                (f"@{stage}/train/dataset/helloworld2.txt", False),
                (f"@{stage}/train/helloworld.txt", False),
                (f"@{stage}/test/testhelloworld.txt", False),
                (f"@{stage}/random", False),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.isdir(path)
                    self.assertEqual(actual, expected)

    def test_isfile(self) -> None:
        """Test if sfcfs can call isfile() to check if a path is a file."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (f"@{stage}/train/dataset/", False),
                (f"@{stage}/train/dataset", False),
                (f"@{stage}/train", False),
                (f"@{stage}/test/", False),
                (f"@{stage}/train/dataset/helloworld1.txt", True),
                (f"@{stage}/train/dataset/helloworld2.txt", True),
                (f"@{stage}/train/helloworld.txt", True),
                (f"@{stage}/test/testhelloworld.txt", True),
                (f"@{stage}/random", False),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.isfile(path)
                    self.assertEqual(actual, expected)

    def test_exists(self) -> None:
        """Test if sfcfs can call exists() to check if a path exists as a file or directory."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (f"@{stage}/train/dataset/", True),
                (f"@{stage}/train/dataset", True),
                (f"@{stage}/train", True),
                (f"@{stage}/test/", True),
                (f"@{stage}/train/dataset/helloworld1.txt", True),
                (f"@{stage}/train/dataset/helloworld2.txt", True),
                (f"@{stage}/train/helloworld.txt", True),
                (f"@{stage}/test/testhelloworld.txt", True),
                (f"@{stage}/random", False),
                (f"@{stage}/test/random", False),
                (f"@{stage}/test/random/", False),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.exists(path)
                    self.assertEqual(actual, expected)

    def test_find(self) -> None:
        """Test if sfcfs can call find() to return all the files under a path."""
        for stage in [self.stage1, self.stage2]:
            for path, expected in [
                (
                    f"@{stage}/train/dataset/",
                    [f"@{stage}/train/dataset/helloworld1.txt", f"@{stage}/train/dataset/helloworld2.txt"],
                ),
                (
                    f"@{stage}/train/",
                    [
                        f"@{stage}/train/dataset/helloworld1.txt",
                        f"@{stage}/train/dataset/helloworld2.txt",
                        f"@{stage}/train/helloworld.txt",
                    ],
                ),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.find(path)
                    self.assertEqual(actual, expected)

    def test_du(self) -> None:
        """Test if sfcfs can call du() to return the total size of files under a path."""
        for stage in [self.stage1, self.stage2]:
            len_of_content = len(self._get_content(stage))
            for path, expected in [
                (f"@{stage}/train/dataset/", 2 * len_of_content),
                (f"@{stage}/train", 3 * len_of_content),
                (f"@{stage}/train/dataset/helloworld1.txt", len_of_content),
            ]:
                for fs in [self.sffs1, self.sffs2]:
                    actual = fs.du(path)
                    self.assertEqual(actual, expected)

    def test_glob(self) -> None:
        """Test if sfsfs can call glob() to do wildcard matching."""
        for stage in [self.stage1, self.stage2]:
            for pattern, expected in [
                (
                    f"@{stage}/train/",
                    [
                        f"@{stage}/train",
                    ],
                ),
                (
                    f"@{stage}/train/*",
                    [
                        f"@{stage}/train/helloworld.txt",
                    ],
                ),
                (
                    f"@{stage}/train/**",
                    [
                        f"@{stage}/train",
                        f"@{stage}/train/dataset/",
                        f"@{stage}/train/dataset/helloworld1.txt",
                        f"@{stage}/train/dataset/helloworld2.txt",
                        f"@{stage}/train/helloworld.txt",
                    ],
                ),
                (
                    f"@{stage}/train/dataset/?elloworld1*",
                    [
                        f"@{stage}/train/dataset/helloworld1.txt",
                    ],
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
                fs.optimize_read(["@ML_DATASETS.public.stage_does_not_exist/aaa"])

    def test_fs_serializability(self) -> None:
        """Test if an object of Snowflake FS can be serialized using pickle."""

        sfcfs_pickle = sfcfs.SFFileSystem(sf_connection=self.sf_connection)

        pickled_data = pickle.dumps(sfcfs_pickle)
        sfcfs_deserialized = pickle.loads(pickled_data)
        assert sfcfs_deserialized._conn is not None


if __name__ == "__main__":
    absltest.main()

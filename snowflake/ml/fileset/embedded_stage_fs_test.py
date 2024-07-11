import os
from typing import Dict, List, cast

import boto3
import requests
from absl.testing import absltest, parameterized
from moto import server

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml.fileset import embedded_stage_fs
from snowflake.ml.test_utils import mock_data_frame, mock_session


class SFEmbeddedStageFileSystemTest(parameterized.TestCase):
    """Testing SFEmbeddedStageFileSystem class."""

    # Arbitrary bucket name and path comes from the mocked GS GET response
    bucket_name = "sfc-dev1"

    urls: Dict[str, str] = {}
    content = b"hello world"
    db = "TESTDB"
    schema = "TESTSCHEMA"
    stage = "TESTSTAGE"

    domain = "dataset"
    name = f"{db}.{schema}.{stage}"

    file1 = "helloworld1"
    file2 = "helloworld2"
    subdir = "mydir"
    file3 = f"{subdir}/helloworld"
    file_list = [file1, file2, file3]

    # Adding trivial AWS creds to satisfy github bazel test
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

    @classmethod
    def setUpClass(cls) -> None:
        """Creates a file system object and put files into it.

        This starts a local S3 server which we'll use to mock a Snowflake stage.
        """
        cls.time_patcher = absltest.mock.patch("time.time", return_value=1)
        cls.mock_time = cls.time_patcher.start()

        # Let the server pick a port.
        cls.server = server.ThreadedMotoServer(port=0)
        cls.server.start()

        cls.mock_connection = absltest.mock.MagicMock(spec=connection.SnowflakeConnection)

        # Setup the backend server
        cls.ENDPOINT_URI = f"http://localhost:{cls.server._server.port}/"
        s3_resource = boto3.resource("s3", "us-east-1", endpoint_url=cls.ENDPOINT_URI)
        s3_client = boto3.client("s3", "us-east-1", endpoint_url=cls.ENDPOINT_URI)
        bucket = s3_resource.Bucket(cls.bucket_name)
        bucket.create()
        for file in cls.file_list:
            fileobj = s3_resource.Object(cls.bucket_name, file)
            fileobj.put(Body=cls.content)
            cls.urls[file] = s3_client.generate_presigned_url(
                "get_object", Params={"Bucket": cls.bucket_name, "Key": file}, ExpiresIn=3600
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.time_patcher.stop()
        cls.server.stop()

    def setUp(self) -> None:
        self.session = mock_session.MockSession(conn=None, test_case=self)

    def _create_new_snowfs(self) -> embedded_stage_fs.SFEmbeddedStageFileSystem:
        stagefs = embedded_stage_fs.SFEmbeddedStageFileSystem(
            domain=self.domain,
            name=self.name,
            snowpark_session=cast(snowpark.Session, self.session),
        )
        return stagefs

    def _mock_collect_res(self, prefix: str, collect_block: bool = True) -> mock_data_frame.MockDataFrame:
        res = []
        for file in self.file_list:
            if file.startswith(prefix):
                res.append(snowpark.Row(name=file, size=10, md5="xx", last_modified="00"))
        return mock_data_frame.MockDataFrame(
            collect_result=(res if collect_block else mock_data_frame.MockAsyncJob(res)),
            collect_block=collect_block,
        )

    def _add_mock_test_case(self, prefix: str) -> None:
        self.session.add_mock_sql(
            query=f"LIST 'snow://{self.domain}/{self.name}/{prefix}'",
            result=self._mock_collect_res(prefix, collect_block=False),
        )

    def _mock_presigned_url_fetcher(self, files: str, lifetime: int = 0) -> List[snowpark.Row]:
        return [snowpark.Row(NAME=file, URL=self.urls[file]) for file in files]

    def test_moto_setup(self) -> None:
        """Test if unittest setUp has already prepared file object in S3 bucket."""
        s3 = boto3.resource("s3", "us-east-1", endpoint_url=self.ENDPOINT_URI)
        for file in self.file_list:
            with self.subTest():
                fileobj = s3.Object(self.bucket_name, file)
                filecontent = fileobj.get()["Body"].read()
                self.assertEqual(filecontent, self.content)

    def test_presigned_url_setup(self) -> None:
        """Test if unittest setUp could mock urls and reach those urls."""
        for file in self.file_list:
            with self.subTest():
                url = self.urls[file]
                response = requests.get(url)
                self.assertEqual(bytes(response.text, "utf-8"), self.content)

    def test_negative_create_new_snowfs(self) -> None:
        """Test if an invalid input of init a stagefs will raise expcetions."""
        with self.assertRaises(ValueError):
            embedded_stage_fs.SFEmbeddedStageFileSystem(
                domain=self.domain,
                name=self.name,
            )
        with self.assertRaises(ValueError):
            embedded_stage_fs.SFEmbeddedStageFileSystem(
                domain=self.domain,
                name=self.name,
                snowpark_session=cast(snowpark.Session, self.session),
                sf_connection=self.mock_connection,
            )

    @parameterized.parameters(  # type: ignore[misc]
        ("", ["helloworld1", "helloworld2", "mydir/"]),
        ("hello", []),
        ("helloworld", []),
        ("helloworld1", ["helloworld1"]),
        ("helloworld2", ["helloworld2"]),
        ("mydir", ["mydir/helloworld"]),
        ("mydir/", ["mydir/helloworld"]),
        ("mydir/helloworld", ["mydir/helloworld"]),
    )
    def test_ls(self, prefix: str, expected_res: List[str]) -> None:
        """Test fsspec overridden method ls() could list objects."""
        self._add_mock_test_case(prefix)
        fs = self._create_new_snowfs()
        self.assertListEqual(fs.ls(prefix), expected_res)

    @parameterized.parameters(  # type: ignore[misc]
        ("helloworld1", [""], True),
        ("helloworld2", [""], True),
        ("mydir/helloworld", ["mydir"], True),
        ("mydir/helloworld", ["", "mydir/helloworld"], False),
        ("mydir/nonexist_file", ["mydir", "mydir/nonexist_file"], False),
    )
    def test_exists(self, file_path: str, mock_prefixes: List[str], expected_res: bool) -> None:
        """Test fsspec method exists() could check if a path exists or not."""
        stagefs = self._create_new_snowfs()
        for prefix in mock_prefixes:
            self._add_mock_test_case(prefix)
        actual = stagefs.exists(file_path)
        self.assertEqual(expected_res, actual)

    @parameterized.parameters(  # type: ignore[misc]
        ("mydir", True),
        ("helloworld1", False),
    )
    def test_isdir(self, path: str, expected: bool) -> None:
        """Test fsspec method isdir() could check if an object is a directory."""
        stagefs = self._create_new_snowfs()
        self._add_mock_test_case("")
        self.assertEqual(expected, stagefs.isdir(path))

    @parameterized.parameters(  # type: ignore[misc]
        ("mydir", False),
        ("helloworld1", True),
    )
    def test_isfile(self, path: str, expected: bool) -> None:
        """Test fsspec method isfile() could check if an object is a file."""
        stagefs = self._create_new_snowfs()
        self._add_mock_test_case("")
        self.assertEqual(expected, stagefs.isfile(path))

    def test_find(self) -> None:
        """Test fsspec method find() could find objects with a given prefix."""
        stagefs = self._create_new_snowfs()
        self._add_mock_test_case("")
        self._add_mock_test_case(self.subdir)
        actual = stagefs.find("")
        self.assertListEqual(self.file_list, actual)

    def test_open(self) -> None:
        """Test fsspec overridden method open() could return a Python file-like object."""
        with absltest.mock.patch.object(
            embedded_stage_fs.SFEmbeddedStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_snowfs()
            for file in self.file_list:
                with self.subTest():
                    fp = stagefs.open(file)
                    self.assertEqual(fp.read(), self.content)

    def test_optimize_read(self) -> None:
        """Test if optimize_read() can refresh all the stored presigned_urls."""
        with absltest.mock.patch.object(
            embedded_stage_fs.SFEmbeddedStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_snowfs()
            stagefs.optimize_read(self.file_list)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 14401)
            self.mock_time.return_value = 10
            stagefs.optimize_read(self.file_list)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 14410)
            self.mock_time.return_value = 1

    def test_open_refresh(self) -> None:
        """Test if fsspec open() could refresh stored presigned_urls when they are expired."""
        with absltest.mock.patch.object(
            embedded_stage_fs.SFEmbeddedStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_snowfs()
            stagefs.optimize_read(self.file_list)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 14401)
            self.mock_time.return_value = 15000
            fp = stagefs.open(self.file1)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 29400)
            self.assertEqual(fp.read(), self.content)
            self.mock_time.return_value = 1

    @parameterized.parameters(  # type: ignore[misc]
        ("versions/my_version/file.ext", "versions/my_version"),
        ("versions/my_version/subdir/file.ext", "versions/my_version/subdir"),
        ("versions/my_version/", "versions/my_version"),
        ("versions/my_version", "versions/my_version"),
        ("versions/my_version//", "versions/my_version"),
        ("versions/my_version//file.ext", "versions/my_version"),
        ("versions/my_version//subdir/file.ext", "versions/my_version//subdir"),
        ("snow://dataset/my_ds/versions/my_version/file.ext", "snow://dataset/my_ds/versions/my_version"),
        ("snow://dataset/my_ds/versions/my_version/subdir/file.ext", "snow://dataset/my_ds/versions/my_version/subdir"),
        ("snow://dataset/my_ds/versions/my_version/", "snow://dataset/my_ds/versions/my_version"),
        ("snow://dataset/my_ds/versions/my_version", "snow://dataset/my_ds/versions/my_version"),
    )
    def test_parent(self, input: str, expected: str) -> None:
        actual = embedded_stage_fs.SFEmbeddedStageFileSystem._parent(input)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    absltest.main()

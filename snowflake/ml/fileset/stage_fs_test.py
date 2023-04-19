import os
from typing import Dict, List

import boto3

# library `requests` has known stubs but is not installed.
# TODO(zpeng): we may need to install as many mypy stubs as possible. However that
# would require installing mypy when initializing the bazel conda environment.
import requests  # type: ignore
import stage_fs
from absl.testing import absltest
from moto import server

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml.test_utils import mock_data_frame, mock_session


class SFStageFileSystemTest(absltest.TestCase):
    """Testing SFStageFileSystem class."""

    # Arbitary bucket name and path comes from the mocked GS GET response
    bucket_name = "sfc-dev1"

    file1 = "helloworld1"
    file2 = "helloworld2"
    subdir = "mydir"
    file3 = "helloworld"
    file_list = [file1, file2, f"{subdir}/{file3}"]
    urls: Dict[str, str] = {}
    content = b"hello world"
    db = "TESTDB"
    schema = "TESTSCHEMA"
    stage = "TESTSTAGE"

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

    def _create_new_stagefs(self) -> stage_fs.SFStageFileSystem:
        stagefs = stage_fs.SFStageFileSystem(
            db=self.db,
            schema=self.schema,
            stage=self.stage,
            snowpark_session=self.session,
        )
        return stagefs

    def _mock_collect_res(self, prefix: str) -> mock_data_frame.MockDataFrame:
        res = []
        for file in self.file_list:
            if file.startswith(prefix):
                res.append(snowpark.Row(name=f"{self.stage}/{file}", size=10, md5="xx", last_modified="00"))
        return mock_data_frame.MockDataFrame(collect_result=res)

    def _add_mock_test_case(self, prefix: str) -> None:
        self.session.add_mock_sql(
            query=f"LIST @{self.db}.{self.schema}.{self.stage}/{prefix}",
            result=self._mock_collect_res(prefix),
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

    def test_negative_create_new_stagefs(self) -> None:
        """Test if an invalid input of init a stagefs will raise expcetions."""
        with self.assertRaises(ValueError):
            stage_fs.SFStageFileSystem(
                db=self.db,
                schema=self.schema,
                stage=self.stage,
            )
        with self.assertRaises(ValueError):
            stage_fs.SFStageFileSystem(
                db=self.db,
                schema=self.schema,
                stage=self.stage,
                snowpark_session=self.session,
                sf_connection=self.mock_connection,
            )

    def test_ls(self) -> None:
        """Test fsspec overrided method ls() could list objects."""
        test_cases = [
            ("", ["helloworld1", "helloworld2", "mydir/"]),
            ("hello", []),
            ("helloworld", []),
            ("helloworld1", ["helloworld1"]),
            ("helloworld2", ["helloworld2"]),
            ("mydir", ["mydir/helloworld"]),
            ("mydir/", ["mydir/helloworld"]),
            ("mydir/helloworld", ["mydir/helloworld"]),
        ]
        for prefix, expected_res in test_cases:
            with self.subTest():
                self._add_mock_test_case(prefix)
                fs = self._create_new_stagefs()
                self.assertListEqual(fs.ls(prefix), expected_res)

    def test_exists(self) -> None:
        """Test fsspec method exists() could check if a path exists or not."""
        test_cases = [
            (self.file1, [""], True),
            (self.file2, [""], True),
            (f"{self.subdir}/{self.file3}", [self.subdir], True),
            (self.file3, ["", self.file3], False),
            (f"{self.subdir}/{self.file1}", [self.subdir, f"{self.subdir}/{self.file1}"], False),
        ]
        stagefs = self._create_new_stagefs()
        for file_path, mock_prefixes, expected_res in test_cases:
            with self.subTest():
                for prefix in mock_prefixes:
                    self._add_mock_test_case(prefix)
                actual = stagefs.exists(file_path)
                self.assertEqual(expected_res, actual)

    def test_isdir(self) -> None:
        """Test fsspec method isdir() could check if an object is a directory."""
        stagefs = self._create_new_stagefs()
        self._add_mock_test_case("")
        self.assertEqual(True, stagefs.isdir(self.subdir))
        self._add_mock_test_case("")
        self.assertEqual(False, stagefs.isdir(self.file1))

    def test_isfile(self) -> None:
        """Test fsspec method isfile() could check if an object is a file."""
        stagefs = self._create_new_stagefs()
        self._add_mock_test_case("")
        self.assertEqual(False, stagefs.isfile(self.subdir))
        self._add_mock_test_case("")
        self.assertEqual(True, stagefs.isfile(self.file1))

    def test_find(self) -> None:
        """Test fsspec method find() could find objects with a given prefix."""
        stagefs = self._create_new_stagefs()
        self._add_mock_test_case("")
        self._add_mock_test_case(self.subdir)
        actual = stagefs.find("")
        self.assertListEqual(self.file_list, actual)

    def test_open(self) -> None:
        """Test fsspec overrided method open() could return a Python file-like object."""
        with absltest.mock.patch.object(
            stage_fs.SFStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_stagefs()
            for file in self.file_list:
                with self.subTest():
                    fp = stagefs.open(file)
                    self.assertEqual(fp.read(), self.content)

    def test_optimize_read(self) -> None:
        """Test if optimize_read() can refresh all the stored presigned_urls."""
        with absltest.mock.patch.object(
            stage_fs.SFStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_stagefs()
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
            stage_fs.SFStageFileSystem, "_fetch_presigned_urls", new=self._mock_presigned_url_fetcher
        ):
            stagefs = self._create_new_stagefs()
            stagefs.optimize_read(self.file_list)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 14401)
            self.mock_time.return_value = 15000
            fp = stagefs.open(self.file1)
            for presigned_url in stagefs._url_cache.values():
                self.assertEqual(presigned_url.expire_at, 29400)
            self.assertEqual(fp.read(), self.content)
            self.mock_time.return_value = 1


if __name__ == "__main__":
    absltest.main()

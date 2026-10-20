import os
import posixpath
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest

from snowflake.connector import errorcode, errors as connector_errors
from snowflake.ml.jobs._interop import data_utils
from snowflake.snowpark import exceptions as sp_exceptions, file_operation


def _mock_session(get_side_effect: Any) -> mock.MagicMock:
    session = mock.MagicMock()
    session.file = mock.create_autospec(file_operation.FileOperation, instance=True)
    session.file.get.side_effect = get_side_effect
    return session


def _write_like_get(path: str, target_directory: str, content: bytes) -> None:
    with open(os.path.join(target_directory, posixpath.basename(path)), "wb") as f:
        f.write(content)


class OpenStreamTest(absltest.TestCase):
    def test_read_stage_file(self) -> None:
        target_directories: list[str] = []

        def fake_get(path: str, target_directory: str, **kwargs: Any) -> list[Any]:
            target_directories.append(target_directory)
            _write_like_get(path, target_directory, b"job result")
            return []

        session = _mock_session(fake_get)

        with data_utils.open_stream("@stage/job/mljob_result", "rb", session=session) as stream:
            # The stream reads the downloaded file in place, so the download directory lives until close.
            self.assertTrue(os.path.exists(os.path.join(target_directories[0], "mljob_result")))
            self.assertEqual(stream.read(), b"job result")
        session.file.get.assert_called_once_with("@stage/job/mljob_result", target_directories[0])
        self.assertFalse(os.path.exists(target_directories[0]))

    def test_concurrent_reads_of_files_with_the_same_name(self) -> None:
        # Every ML Job writes its result to a file named mljob_result.
        contents = {
            "@stage_a/job_a/mljob_result": b"result of job a",
            "@stage_b/job_b/mljob_result": b"result of job b",
        }
        both_downloaded = threading.Barrier(len(contents), timeout=10)

        def fake_get(path: str, target_directory: str, **kwargs: Any) -> list[Any]:
            _write_like_get(path, target_directory, contents[path])
            both_downloaded.wait()
            return []

        session = _mock_session(fake_get)
        read: dict[str, bytes] = {}

        def read_stage_file(path: str) -> None:
            with data_utils.open_stream(path, "rb", session=session) as stream:
                read[path] = stream.read()

        threads = [threading.Thread(target=read_stage_file, args=(path,)) for path in contents]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(read, contents)

    def test_download_error_raises_snowpark_sql_exception(self) -> None:
        file_not_found = connector_errors.OperationalError(
            msg="the file does not exist.", errno=errorcode.ER_FILE_NOT_EXISTS
        )
        target_directories: list[str] = []

        def failing_get(path: str, target_directory: str, **kwargs: Any) -> list[Any]:
            target_directories.append(target_directory)
            raise file_not_found

        session = _mock_session(failing_get)

        try:
            data_utils.open_stream("@stage/job/mljob_result", "rb", session=session)
        except sp_exceptions.SnowparkSQLException as e:
            self.assertIs(e.conn_error, file_not_found)
            # Checked while the exception, and the frame it references, is still alive.
            self.assertFalse(os.path.exists(target_directories[0]))
        else:
            self.fail("open_stream did not raise")


if __name__ == "__main__":
    absltest.main()

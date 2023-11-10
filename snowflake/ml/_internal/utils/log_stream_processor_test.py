import logging
from io import StringIO

from absl.testing import absltest

from snowflake.ml._internal.utils import log_stream_processor


class LogStreamProcessorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self.log_handler)

    def tearDown(self) -> None:
        logging.getLogger().removeHandler(self.log_handler)
        self.log_stream.close()
        logging.shutdown()

    def reset_log_stream(self) -> None:
        # Clear the log stream
        self.log_stream.truncate(0)
        self.log_stream.seek(0)

    def test_only_new_log_is_shown(self) -> None:
        lsp = log_stream_processor.LogStreamProcessor()
        log1 = "TIMESTAMP1: HI 1"
        log2 = "TIMESTAMP1: HI 1 \n TIMESTAMP2: HI 2"
        log3 = "TIMESTAMP1: HI 1 \n TIMESTAMP2: HI 2 \n TIMESTAMP3: HI 3"
        log4 = "TIMESTAMP1: HI 1 \n TIMESTAMP2: HI 2 \n TIMESTAMP3: HI 3"

        lsp.process_new_logs(log1)
        self.assertEqual("TIMESTAMP1: HI 1", self.log_stream.getvalue().strip())

        self.reset_log_stream()

        lsp.process_new_logs(log2)
        self.assertEqual("TIMESTAMP2: HI 2", self.log_stream.getvalue().strip())

        self.reset_log_stream()

        lsp.process_new_logs(log3)
        self.assertEqual("TIMESTAMP3: HI 3", self.log_stream.getvalue().strip())

        self.reset_log_stream()

        # No new log returned
        lsp.process_new_logs(log4)
        self.assertEqual("", self.log_stream.getvalue().strip())

        self.reset_log_stream()

        # Process empty log
        lsp.process_new_logs(None)
        self.assertEqual("", self.log_stream.getvalue().strip())


if __name__ == "__main__":
    absltest.main()

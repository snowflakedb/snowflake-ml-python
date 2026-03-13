from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.tee import OutputTee


class OutputTeeTest(absltest.TestCase):
    """Tests for OutputTee class."""

    def setUp(self) -> None:
        self.stream1 = MagicMock()
        self.stream2 = MagicMock()
        self.stream1.writable.return_value = True
        self.stream2.writable.return_value = True
        self.tee = OutputTee(self.stream1, self.stream2)

    def test_flush(self) -> None:
        """Test that flushing OutputTee flushes all streams."""
        self.tee.flush()
        self.stream1.flush.assert_called_once()
        self.stream2.flush.assert_called_once()

    def test_writable(self) -> None:
        """Test that OutputTee is writable if and only if all streams are writable."""
        self.assertTrue(self.tee.writable())
        self.stream2.writable.return_value = False
        self.assertFalse(self.tee.writable())

    def test_write(self) -> None:
        """Test that writing to OutputTee writes to all streams."""
        data = "test data"
        self.assertEqual(self.tee.write(data), len(data))
        self.stream1.write.assert_called_once_with(data)
        self.stream2.write.assert_called_once_with(data)

    def test_writelines(self) -> None:
        """Test that writelines writes to all streams."""
        lines = ["line 1\n", "line 2\n"]
        self.tee.writelines(lines)
        self.stream1.writelines.assert_called_once_with(lines)
        self.stream2.writelines.assert_called_once_with(lines)


if __name__ == "__main__":
    absltest.main()

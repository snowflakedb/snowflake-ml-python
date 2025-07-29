import logging
import os
import tempfile
import unittest.mock as mock

from absl.testing import absltest

from snowflake.ml._internal.utils import service_logger


class ServiceLoggerTest(absltest.TestCase):
    """Test service_logger functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_operation_id = "test_operation_12345"
        self.test_logger_name = "test_logger"
        self.test_color = service_logger.LogColor.BLUE

        # Store original root logger level
        self._original_root_level = logging.getLogger().level

        # Clear any existing loggers to avoid test interference
        self._clear_all_loggers()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Restore original root logger level
        logging.getLogger().setLevel(self._original_root_level)

        # Clear loggers created during tests
        self._clear_all_loggers()

    def _clear_all_loggers(self) -> None:
        """Clear all loggers and handlers to ensure test isolation."""
        # Clear root logger handlers
        logging.getLogger().handlers.clear()

        # Clear all loggers that might interfere with tests
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("snowflake_ml_operation_") or name.startswith("test_"):
                logger = logging.getLogger(name)
                # Close and remove all handlers
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
                logger.handlers.clear()
                # Remove from manager
                if name in logging.Logger.manager.loggerDict:
                    del logging.Logger.manager.loggerDict[name]

    def test_get_logger_without_operation_id(self) -> None:
        """Test that get_logger works without operation_id (backward compatibility)."""
        # Set root logger to INFO level to ensure handler is added
        logging.getLogger().setLevel(logging.INFO)

        logger = service_logger.get_logger(self.test_logger_name, self.test_color)

        # Should have one handler (StreamHandler) when in verbose mode
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

        # Should not have a parent logger (beyond the root logger)
        root_logger = logging.getLogger()
        self.assertEqual(logger.parent, root_logger)

        # Should use custom formatter
        self.assertIsInstance(logger.handlers[0].formatter, service_logger.CustomFormatter)

    def test_get_logger_without_operation_id_quiet_mode(self) -> None:
        """Test that get_logger without operation_id doesn't add handler in quiet mode."""
        # Set root logger to WARNING level (quiet mode)
        logging.getLogger().setLevel(logging.WARNING)

        logger = service_logger.get_logger(self.test_logger_name, self.test_color)

        # Should have no handlers in quiet mode
        self.assertEqual(len(logger.handlers), 0)

    def test_get_logger_with_operation_id(self) -> None:
        """Test that get_logger creates parent logger with operation_id in verbose mode."""
        # Set root logger to INFO level to ensure handler is added
        logging.getLogger().setLevel(logging.INFO)

        logger = service_logger.get_logger(self.test_logger_name, self.test_color, self.test_operation_id)

        # Should have one handler (StreamHandler) in verbose mode
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

        # Should have parent logger
        self.assertIsNotNone(logger.parent)
        if logger.parent:
            self.assertEqual(logger.parent.name, f"snowflake_ml_operation_{self.test_operation_id}")

        # Should propagate to parent
        self.assertTrue(logger.propagate)

    def test_get_logger_with_operation_id_quiet_mode(self) -> None:
        """Test that get_logger with operation_id doesn't add console handler in quiet mode."""
        # Set root logger to WARNING level (quiet mode)
        logging.getLogger().setLevel(logging.WARNING)

        logger = service_logger.get_logger(self.test_logger_name, self.test_color, self.test_operation_id)

        # Should have no console handlers in quiet mode (only file via propagation)
        self.assertEqual(len(logger.handlers), 0)

        # Should still have parent logger for file output
        self.assertIsNotNone(logger.parent)
        if logger.parent:
            self.assertEqual(logger.parent.name, f"snowflake_ml_operation_{self.test_operation_id}")

        # Should still propagate to parent for file logging
        self.assertTrue(logger.propagate)

    def test_parent_logger_creation_with_file(self) -> None:
        """Test that parent logger is created with FileHandler when filesystem is writable."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
            log_file_path = "/fake/writable/path/test_operation.log"
            mock_path.return_value = log_file_path

            # Mock FileHandler to avoid actual file creation
            with mock.patch("logging.FileHandler") as mock_file_handler:
                mock_handler_instance = mock.MagicMock()
                mock_handler_instance.level = logging.DEBUG  # Set proper level for logging comparisons
                mock_handler_instance.close = mock.MagicMock()  # Add close method for cleanup
                mock_file_handler.return_value = mock_handler_instance

                parent_logger = service_logger._get_or_create_parent_logger(self.test_operation_id)

                # Should have correct name
                self.assertEqual(parent_logger.name, f"snowflake_ml_operation_{self.test_operation_id}")

                # Should have one handler (mocked FileHandler)
                self.assertEqual(len(parent_logger.handlers), 1)
                self.assertEqual(parent_logger.handlers[0], mock_handler_instance)

                # Should not propagate to root logger
                self.assertFalse(parent_logger.propagate)

                # Should be set to DEBUG level
                self.assertEqual(parent_logger.level, logging.DEBUG)

    def test_parent_logger_creation_readonly_filesystem(self) -> None:
        """Test that parent logger works without FileHandler when filesystem is readonly."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
            # Simulate readonly filesystem by returning None
            mock_path.return_value = None

            parent_logger = service_logger._get_or_create_parent_logger(self.test_operation_id)

            # Should have correct name
            self.assertEqual(parent_logger.name, f"snowflake_ml_operation_{self.test_operation_id}")

            # Should have no handlers (console-only logging)
            self.assertEqual(len(parent_logger.handlers), 0)

            # Should not propagate to root logger
            self.assertFalse(parent_logger.propagate)

            # Should be set to DEBUG level
            self.assertEqual(parent_logger.level, logging.DEBUG)

    def test_parent_logger_creation_file_handler_fails(self) -> None:
        """Test parent logger fallback when FileHandler creation fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, f"{self.test_operation_id}.log")

            with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
                mock_path.return_value = log_file_path

                with mock.patch("logging.FileHandler", side_effect=OSError("Permission denied")):
                    parent_logger = service_logger._get_or_create_parent_logger(self.test_operation_id)

                    # Should have correct name
                    self.assertEqual(parent_logger.name, f"snowflake_ml_operation_{self.test_operation_id}")

                    # Should have no handlers due to FileHandler failure
                    self.assertEqual(len(parent_logger.handlers), 0)

                    # Should not propagate to root logger
                    self.assertFalse(parent_logger.propagate)

                    # Should be set to DEBUG level
                    self.assertEqual(parent_logger.level, logging.DEBUG)

    def test_parent_logger_reuse(self) -> None:
        """Test that parent logger is reused for same operation_id."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
            log_file_path = "/fake/writable/path/test_operation.log"
            mock_path.return_value = log_file_path

            # Mock FileHandler to avoid actual file creation
            with mock.patch("logging.FileHandler") as mock_file_handler:
                mock_handler_instance = mock.MagicMock()
                mock_handler_instance.level = logging.DEBUG  # Set proper level for logging comparisons
                mock_handler_instance.close = mock.MagicMock()  # Add close method for cleanup
                mock_file_handler.return_value = mock_handler_instance

                parent_logger_1 = service_logger._get_or_create_parent_logger(self.test_operation_id)
                parent_logger_2 = service_logger._get_or_create_parent_logger(self.test_operation_id)

                # Should be the same instance
                self.assertIs(parent_logger_1, parent_logger_2)

                # Should still have only one handler
                self.assertEqual(len(parent_logger_1.handlers), 1)

    def test_test_writability_success(self) -> None:
        """Test _test_writability returns True for writable directory."""
        # Mock the file operations to avoid actual file writing
        with mock.patch("os.makedirs"), mock.patch("builtins.open"), mock.patch("os.remove"):
            result = service_logger._test_writability("/fake/writable/dir")
            self.assertTrue(result)

    def test_test_writability_failure(self) -> None:
        """Test _test_writability returns False for non-writable directory."""
        with mock.patch("builtins.open", side_effect=OSError("Permission denied")):
            result = service_logger._test_writability("/fake/readonly/dir")
            self.assertFalse(result)

    def test_try_log_location_success(self) -> None:
        """Test _try_log_location returns path when directory is writable."""
        # Mock _test_writability to avoid actual file writing
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability", return_value=True):
            test_dir = "/fake/writable/dir"
            result = service_logger._try_log_location(test_dir, self.test_operation_id)
            expected_path = os.path.join(test_dir, f"{self.test_operation_id}.log")
            self.assertEqual(result, expected_path)

    def test_try_log_location_failure(self) -> None:
        """Test _try_log_location returns None when directory is not writable."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability", return_value=False):
            result = service_logger._try_log_location("/fake/readonly/dir", self.test_operation_id)
            self.assertIsNone(result)

    def test_get_log_file_path_success(self) -> None:
        """Test _get_log_file_path returns path when first location is writable."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability", return_value=True):
            with mock.patch("platformdirs.user_log_dir", return_value="/test/log/dir"):
                result = service_logger._get_log_file_path(self.test_operation_id)
                expected_path = os.path.join("/test/log/dir", f"{self.test_operation_id}.log")
                self.assertEqual(result, expected_path)

    def test_get_log_file_path_fallback_to_temp(self) -> None:
        """Test _get_log_file_path falls back to temp directory."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability") as mock_test:
            # First call (platformdirs) fails, second call (temp dir) succeeds
            mock_test.side_effect = [False, True]

            with mock.patch("platformdirs.user_log_dir", return_value="/readonly/log/dir"):
                with mock.patch("tempfile.gettempdir", return_value="/tmp"):
                    result = service_logger._get_log_file_path(self.test_operation_id)
                    expected_path = os.path.join("/tmp", "snowflake-ml-logs", f"{self.test_operation_id}.log")
                    self.assertEqual(result, expected_path)

    def test_get_log_file_path_fallback_to_cwd(self) -> None:
        """Test _get_log_file_path falls back to current working directory."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability") as mock_test:
            # First two calls fail, third call (cwd) succeeds
            mock_test.side_effect = [False, False, True]

            result = service_logger._get_log_file_path(self.test_operation_id)
            expected_path = os.path.join(".", f"{self.test_operation_id}.log")
            self.assertEqual(result, expected_path)

    def test_get_log_file_path_all_locations_fail(self) -> None:
        """Test _get_log_file_path returns None when all locations are readonly."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._test_writability", return_value=False):
            result = service_logger._get_log_file_path(self.test_operation_id)
            self.assertIsNone(result)

    def test_get_log_file_location_success(self) -> None:
        """Test get_log_file_location function returns path when available."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
            expected_path = "/test/path/test_operation.log"
            mock_path.return_value = expected_path

            result = service_logger.get_log_file_location(self.test_operation_id)

            mock_path.assert_called_once_with(self.test_operation_id)
            self.assertEqual(result, expected_path)

    def test_get_log_file_location_readonly_filesystem(self) -> None:
        """Test get_log_file_location returns None when filesystem is readonly."""
        with mock.patch("snowflake.ml._internal.utils.service_logger._get_log_file_path") as mock_path:
            mock_path.return_value = None

            result = service_logger.get_log_file_location(self.test_operation_id)

            mock_path.assert_called_once_with(self.test_operation_id)
            self.assertIsNone(result)


if __name__ == "__main__":
    absltest.main()

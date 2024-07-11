# Error code from Snowflake Python Connector.
ERRNO_OBJECT_ALREADY_EXISTS = 2002
ERRNO_OBJECT_NOT_EXIST = 2043
ERRNO_FILES_ALREADY_EXISTING = 1030
ERRNO_VERSION_ALREADY_EXISTS = 92917
ERRNO_DATASET_NOT_EXIST = 399019
ERRNO_DATASET_VERSION_NOT_EXIST = 399012
ERRNO_DATASET_VERSION_ALREADY_EXISTS = 399020


class DatasetError(Exception):
    """Base class for other exceptions."""


class DatasetNotExistError(DatasetError):
    """Raised when the requested Dataset does not exist."""


class DatasetExistError(DatasetError):
    """Raised when there is already an existing Dataset with the same name and version in selected schema."""


class DatasetCannotDeleteError(DatasetError):
    """Raised when a Dataset is unable to get deleted."""

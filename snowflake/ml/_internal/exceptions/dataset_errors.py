# Error code from Snowflake Python Connector.
ERRNO_FILE_EXIST_IN_STAGE = "001030"
ERRNO_DOMAIN_NOT_EXIST = "002003"
ERRNO_STAGE_NOT_EXIST = "391707"


class DatasetError(Exception):
    """Base class for other exceptions."""


class DatasetNotExistError(DatasetError):
    """Raised when the requested Dataset does not exist."""


class DatasetExistError(DatasetError):
    """Raised when there is already an existing Dataset with the same name and version in selected schema."""


class DatasetLocationError(DatasetError):
    """Raised when the given location to the Dataset is invalid."""


class DatasetCannotDeleteError(DatasetError):
    """Raised when a Dataset is unable to get deleted."""


class DatasetIntegrityError(DatasetError):
    """Raised when the Dataset contains invalid or unrecognized files."""


class DatasetInvalidSourceError(DatasetError, ValueError):
    """Raised when trying to create a Dataset from an invalid data source"""

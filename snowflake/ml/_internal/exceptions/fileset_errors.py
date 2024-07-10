# Error code from Snowflake Python Connector.
ERRNO_FILE_EXIST_IN_STAGE = 1030
ERRNO_DOMAIN_NOT_EXIST = 2003
ERRNO_STAGE_NOT_EXIST = 391707


class FileSetError(Exception):
    """Base class for other exceptions."""


class FileSetExistError(FileSetError):
    """Raised when there is already existing fileset with the same name in selected stage."""


class FileSetLocationError(FileSetError):
    """Raised when the given location to the fileset is invalid."""


class FileSetCannotDeleteError(FileSetError):
    """Raised when a FileSet is unable to get deleted."""


class FileSetAlreadyDeletedError(FileSetError):
    """Raised when any method is called from an already deleted fileset instance."""


class StageNotFoundError(FileSetError):
    """Raised when the target stage doesn't exist."""


class StageFileNotFoundError(FileSetError):
    """Raised when the target stage file doesn't exist."""


class MoreThanOneQuerySourceError(FileSetError):
    """Raised when the files of a FileSet come from different queries."""

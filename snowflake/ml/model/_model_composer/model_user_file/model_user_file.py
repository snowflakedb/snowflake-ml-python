import os
import pathlib

from snowflake.ml._internal import file_utils


class ModelUserFile:
    """Class representing a user provided file.

    Attributes:
        subdirectory_name: A local path where model related files should be dumped to.
        local_path: A list of ModelRuntime objects managing the runtimes and environment in the MODEL object.
    """

    USER_FILES_DIR_REL_PATH = "user_files"

    def __init__(self, subdirectory_name: pathlib.PurePosixPath, local_path: pathlib.Path) -> None:
        self.subdirectory_name = subdirectory_name
        self.local_path = local_path

    def save(self, workspace_path: pathlib.Path) -> str:
        user_files_path = workspace_path / ModelUserFile.USER_FILES_DIR_REL_PATH / self.subdirectory_name
        user_files_path.mkdir(parents=True, exist_ok=True)

        # copy the file to the workspace
        file_utils.copy_file_or_tree(str(self.local_path), str(user_files_path))
        return os.path.join(self.subdirectory_name, self.local_path.name)

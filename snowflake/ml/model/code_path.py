"""CodePath class for selective code packaging in model registry."""

import os
from dataclasses import dataclass
from typing import Optional

_ERR_ROOT_NOT_FOUND = "CodePath: root '{root}' does not exist (resolved to {resolved})."
_ERR_WILDCARDS_NOT_SUPPORTED = "CodePath: Wildcards are not supported in filter. Got '{filter}'. Use exact paths only."
_ERR_FILTER_MUST_BE_RELATIVE = "CodePath: filter must be a relative path, got absolute path '{filter}'."
_ERR_FILTER_HOME_PATH = "CodePath: filter must be a relative path, got home directory path '{filter}'."
_ERR_FILTER_ON_FILE_ROOT = (
    "CodePath: cannot apply filter to a file root. " "Root '{root}' is a file. Use filter only with directory roots."
)
_ERR_FILTER_ESCAPES_ROOT = (
    "CodePath: filter '{filter}' escapes root directory '{root}'. " "Relative paths must stay within root."
)
_ERR_FILTER_NOT_FOUND = "CodePath: filter '{filter}' under root '{root}' does not exist (resolved to {resolved})."


@dataclass(frozen=True)
class CodePath:
    """Specifies a code path with optional filtering for selective inclusion.

    Args:
        root: The root directory or file path (absolute or relative to cwd).
        filter: Optional relative path under root to select a subdirectory or file.
            The filter also determines the destination path under code/.

    Examples:
        CodePath("project/src/")                          # Copy entire src/ to code/src/
        CodePath("project/src/", filter="utils")          # Copy utils/ to code/utils/
        CodePath("project/src/", filter="lib/helpers")    # Copy to code/lib/helpers/
    """

    root: str
    filter: Optional[str] = None

    def __post_init__(self) -> None:
        if self.filter == "":
            object.__setattr__(self, "filter", None)

    def __repr__(self) -> str:
        if self.filter:
            return f"CodePath({self.root!r}, filter={self.filter!r})"
        return f"CodePath({self.root!r})"

    def _validate_filter(self) -> Optional[str]:
        """Validate and normalize filter, returning normalized filter or None.

        Returns:
            Normalized filter path, or None if no filter is set.

        Raises:
            ValueError: If filter contains wildcards or is an absolute path.
        """
        if self.filter is None:
            return None

        if any(c in self.filter for c in ["*", "?", "[", "]"]):
            raise ValueError(_ERR_WILDCARDS_NOT_SUPPORTED.format(filter=self.filter))

        if self.filter.startswith("~"):
            raise ValueError(_ERR_FILTER_HOME_PATH.format(filter=self.filter))

        filter_normalized = os.path.normpath(self.filter)

        if os.path.isabs(filter_normalized):
            raise ValueError(_ERR_FILTER_MUST_BE_RELATIVE.format(filter=self.filter))

        return filter_normalized

    def _resolve(self) -> tuple[str, str]:
        """Resolve the source path and destination path.

        Returns:
            Tuple of (source_path, destination_relative_path)

        Raises:
            FileNotFoundError: If root or filter path does not exist.
            ValueError: If filter is invalid (wildcards, absolute, escapes root, or applied to file).
        """
        filter_normalized = self._validate_filter()
        root_normalized = os.path.normpath(os.path.abspath(self.root))

        if filter_normalized is None:
            if not os.path.exists(root_normalized):
                raise FileNotFoundError(_ERR_ROOT_NOT_FOUND.format(root=self.root, resolved=root_normalized))
            return root_normalized, os.path.basename(root_normalized)

        if not os.path.exists(root_normalized):
            raise FileNotFoundError(_ERR_ROOT_NOT_FOUND.format(root=self.root, resolved=root_normalized))

        if os.path.isfile(root_normalized):
            raise ValueError(_ERR_FILTER_ON_FILE_ROOT.format(root=self.root))

        source = os.path.normpath(os.path.join(root_normalized, filter_normalized))

        if not (source.startswith(root_normalized + os.sep) or source == root_normalized):
            raise ValueError(_ERR_FILTER_ESCAPES_ROOT.format(filter=self.filter, root=self.root))

        if not os.path.exists(source):
            raise FileNotFoundError(_ERR_FILTER_NOT_FOUND.format(filter=self.filter, root=self.root, resolved=source))

        return source, filter_normalized

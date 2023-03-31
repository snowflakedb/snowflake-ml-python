import os
import shutil


def copy_file_or_tree(src: str, dst_dir: str) -> None:
    """Copy file or directory into target directory.

    Args:
        src (str): Source file or directory path.
        dst_dir (str): Destination directory path.
    """
    if os.path.isfile(src):
        shutil.copy(src=src, dst=dst_dir)
    else:
        dir_name = os.path.basename(os.path.abspath(src))
        dst_path = os.path.join(dst_dir, dir_name)
        shutil.copytree(src=src, dst=dst_path, ignore=shutil.ignore_patterns("__pycache__"))

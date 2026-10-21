"""Bootstrap helper — sets up sys.path so shared lib is importable."""

import os
import sys


def setup_lib_path(caller_file: str) -> None:
    """Add the correct lib/ directory to sys.path."""
    here = os.path.dirname(os.path.abspath(caller_file))
    lib_deployed = os.path.join(here, "..", "lib")
    lib_local = os.path.join(here, "..", "..", "..", "lib")

    lib_dir = lib_deployed if os.path.isdir(lib_deployed) else lib_local

    lib_dir = os.path.normpath(lib_dir)
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

"""SparseCam4D preprocessing pipeline package."""
import importlib.util
import sys


def load_submodule(module_name: str, file_path: str):
    """Load a Python module by absolute file path, caching in sys.modules.

    Avoids polluting sys.path with submodule directories.
    Returns the loaded module object.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # register before exec to handle circular refs
    spec.loader.exec_module(mod)
    return mod

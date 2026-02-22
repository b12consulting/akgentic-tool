from importlib import import_module
from typing import Any


def import_class(class_path: str) -> type[Any]:
    """Dynamically import a class from the given path.

    Args:
        class_path: Full module path and class name (e.g., "module.ClassName").

    Returns:
        The imported class.

    Raises:
        ImportError: If module cannot be imported.
        AttributeError: If class not found in module.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]

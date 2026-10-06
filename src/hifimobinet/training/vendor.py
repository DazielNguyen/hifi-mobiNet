"""Import unmodified reference packages from ``vendor/`` without editing them.

``vendor/`` is deliberately outside the installed package, so training is
supported from a source checkout only (``pip install -e .``). Each vendored
package keeps its own relative imports; it is loaded under a private module
name so it can never shadow or be shadowed by another ``vits`` package.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_PACKAGES = {
    "piper_vits": ("piper", "vits"),
    "banhmi": ("banhmi",),
}
_PREFIX = "hifimobinet_vendor_"


def repository_root() -> Path:
    """Checkout root: HIFIMOBINET_HOME if set, else the tree containing this file."""
    env = os.environ.get("HIFIMOBINET_HOME")
    root = Path(env) if env else Path(__file__).resolve().parents[3]
    if not (root / "vendor").is_dir():
        raise RuntimeError(
            f"vendor/ not found under {root}; training requires an editable source checkout "
            "(or HIFIMOBINET_HOME pointing at one)"
        )
    return root


def vendor_package(name: str) -> ModuleType:
    """Load vendor package ``name`` (see _PACKAGES) once and return it."""
    if name not in _PACKAGES:
        raise KeyError(f"Unknown vendor package {name!r}; expected one of {sorted(_PACKAGES)}")
    module_name = _PREFIX + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repository_root().joinpath("vendor", *_PACKAGES[name])
    spec = importlib.util.spec_from_file_location(
        module_name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def vendor_module(name: str, submodule: str) -> ModuleType:
    """Import ``submodule`` (dotted, relative to the vendor package root)."""
    vendor_package(name)
    return importlib.import_module(f"{_PREFIX}{name}.{submodule}")

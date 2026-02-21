import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)
_src_pkg = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "src", "interpretable_genre")
)
if os.path.isdir(_src_pkg) and _src_pkg not in __path__:
    __path__.append(_src_pkg)

"""Import hook making ``import matplotlib`` resolve to ``matplotlib_rs``."""
import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "matplotlib"
_TARGET = "matplotlib_rs"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        tname = _TARGET + fullname[len(_PREFIX):]
        try:
            tspec = importlib.util.find_spec(tname)
        except (ImportError, ValueError):
            tspec = None
        if tspec is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, _AliasLoader(tname),
                                               is_package=tspec.submodule_search_locations is not None)
        return spec


_finder = _Finder()


def install():
    """Make ``import matplotlib`` (and submodules) use matplotlib_rs."""
    if _finder not in sys.meta_path:
        for name in list(sys.modules):
            if name == _PREFIX or name.startswith(_PREFIX + "."):
                mod = sys.modules[name]
                if not getattr(mod, "__name__", "").startswith(_TARGET):
                    raise RuntimeError(
                        "matplotlib was already imported; call matplotlib_rs.install() "
                        "before importing matplotlib")
        sys.meta_path.insert(0, _finder)
        sys.modules[_PREFIX] = sys.modules[_TARGET]


def uninstall():
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    for name in list(sys.modules):
        if name == _PREFIX or name.startswith(_PREFIX + "."):
            if getattr(sys.modules[name], "__name__", "").startswith(_TARGET):
                del sys.modules[name]

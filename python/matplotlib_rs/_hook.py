"""Import hook making ``import matplotlib`` resolve to ``matplotlib_rs`` (and
``import mpl_toolkits`` to its vendored copy, ``matplotlib_rs.mpl_toolkits``)."""
import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "matplotlib"
_TARGET = "matplotlib_rs"
# (imported name prefix, replacement)
_ALIASES = (("matplotlib", "matplotlib_rs"), ("mpl_toolkits", "matplotlib_rs.mpl_toolkits"))


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix, repl in _ALIASES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                tname = repl + fullname[len(prefix):]
                break
        else:
            return None
        try:
            tspec = importlib.util.find_spec(tname)
        except (ImportError, ValueError):
            tspec = None
        if tspec is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, _AliasLoader(tname),
                                               is_package=tspec.submodule_search_locations is not None)
        return spec


class _FallbackFinder(_Finder):
    """Resolves matplotlib's pure-Python dependencies that are not installed
    (consulted last) to the copies vendored in ``matplotlib_rs._vendor``, so
    that scripts written for matplotlib can import them."""

    def find_spec(self, fullname, path=None, target=None):
        for prefix, repl in _FALLBACKS:
            if fullname == prefix or fullname.startswith(prefix + "."):
                tname = repl + fullname[len(prefix):]
                break
        else:
            return None
        tspec = importlib.util.find_spec(tname)
        if tspec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(tname), is_package=tspec.submodule_search_locations is not None)


_FALLBACKS = (("cycler", "matplotlib_rs._vendor.cycler"),
              ("dateutil", "matplotlib_rs._vendor.dateutil"))
_finder = _Finder()
_fallback_finder = _FallbackFinder()


def install():
    """Make ``import matplotlib`` (and submodules) use matplotlib_rs."""
    if _finder not in sys.meta_path:
        for name in list(sys.modules):
            for prefix, _ in _ALIASES:
                if name == prefix or name.startswith(prefix + "."):
                    mod = sys.modules[name]
                    if not getattr(mod, "__name__", "").startswith(_TARGET):
                        raise RuntimeError(
                            f"{prefix} was already imported; call matplotlib_rs.install() "
                            f"before importing {prefix}")
        sys.meta_path.insert(0, _finder)
        sys.meta_path.append(_fallback_finder)
        sys.modules[_PREFIX] = sys.modules[_TARGET]


def uninstall():
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    if _fallback_finder in sys.meta_path:
        sys.meta_path.remove(_fallback_finder)
    for name in list(sys.modules):
        for prefix, _ in _ALIASES:
            if name == prefix or name.startswith(prefix + "."):
                if getattr(sys.modules[name], "__name__", "").startswith(_TARGET):
                    del sys.modules[name]

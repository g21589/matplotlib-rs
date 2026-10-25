"""Backends.  matplotlib_rs renders everything with its Rust Agg port; the
interactive backend names are accepted for compatibility."""


class _BackendRegistry:
    _builtin = ["gtk3agg", "gtk3cairo", "gtk4agg", "gtk4cairo", "macosx", "nbagg",
                "notebook", "qtagg", "qtcairo", "qt5agg", "qt5cairo", "tkagg",
                "tkcairo", "webagg", "wx", "wxagg", "wxcairo", "agg", "cairo",
                "pdf", "pgf", "ps", "svg", "template", "inline", "ipympl", "widget"]

    def is_valid_backend(self, backend):
        if not isinstance(backend, str):
            return False
        b = backend.lower()
        return b in self._builtin or backend.startswith("module://")

    def list_all(self):
        return list(self._builtin)

    def list_builtin(self, filter_=None):
        return list(self._builtin)

    def resolve_backend(self, backend):
        return (backend.lower(), None)

    def resolve_gui_or_backend(self, gui_or_backend):
        return (gui_or_backend.lower(), None)


backend_registry = _BackendRegistry()

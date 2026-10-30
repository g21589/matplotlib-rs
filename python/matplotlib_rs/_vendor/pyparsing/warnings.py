# Vendored by tools/vendor_deps.py from pyparsing 3.3.3 (see licenses/LICENSE_PYPARSING).
class PyparsingWarning(UserWarning):
    """Base warning class for all pyparsing warnings"""


class PyparsingDeprecationWarning(PyparsingWarning, DeprecationWarning):
    """Base warning class for all pyparsing deprecation warnings"""


class PyparsingDiagnosticWarning(PyparsingWarning):
    """Base warning class for all pyparsing diagnostic warnings"""

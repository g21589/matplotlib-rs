"""
The kiwisolver API used by constrained layout, with the Rust port of kiwi's
Cassowary solver (licenses/LICENSE_KIWISOLVER), so that matplotlib_rs needs
only NumPy.

The arithmetic follows kiwisolver's ``symbolics.h`` operation by operation
(e.g. dividing by ``d`` multiplies by ``1.0 / d``) and constraints are reduced
as by kiwi.  kiwi orders the variables of each constraint by their memory
addresses, so its results can change in the last bits from one process to
another; here they are ordered by creation, so that layouts are deterministic
(and in practice identical to kiwisolver's).  Set ``MPLRS_KIWI=kiwisolver``
to use an installed kiwisolver instead (see `load`).
"""

import itertools
import numbers

from . import _mplrs

__all__ = ["BadRequiredStrength", "Constraint", "DuplicateConstraint",
           "DuplicateEditVariable", "Expression", "Solver", "Term",
           "UnknownConstraint", "UnknownEditVariable", "UnsatisfiableConstraint",
           "Variable", "strength"]


def load():
    """This module, or kiwisolver if ``MPLRS_KIWI=kiwisolver``."""
    import os
    import sys
    if os.environ.get("MPLRS_KIWI") == "kiwisolver":
        import kiwisolver
        return kiwisolver
    return sys.modules[__name__]


class BadRequiredStrength(Exception):
    pass


class DuplicateConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint


class DuplicateEditVariable(Exception):
    __slots__ = ("edit_variable",)

    def __init__(self, edit_variable):
        self.edit_variable = edit_variable


class UnknownConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint


class UnknownEditVariable(Exception):
    __slots__ = ("edit_variable",)

    def __init__(self, edit_variable):
        self.edit_variable = edit_variable


class UnsatisfiableConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint


class strength:
    """Constraint strengths (``kiwi::strength``)."""

    @staticmethod
    def create(a, b, c, w=1.0):
        result = 0.0
        result += max(0.0, min(1000.0, a * w)) * 1000000.0
        result += max(0.0, min(1000.0, b * w)) * 1000.0
        result += max(0.0, min(1000.0, c * w))
        return result


strength.required = strength.create(1000.0, 1000.0, 1000.0)
strength.strong = strength.create(1.0, 0.0, 0.0)
strength.medium = strength.create(0.0, 1.0, 0.0)
strength.weak = strength.create(0.0, 0.0, 1.0)

_STRENGTHS = {"required": strength.required, "strong": strength.strong,
              "medium": strength.medium, "weak": strength.weak}
_OPS = {"<=": 0, ">=": 1, "==": 2}


def _clip(value):
    return max(0.0, min(strength.required, value))


def _to_strength(value):
    if isinstance(value, str):
        try:
            return _STRENGTHS[value]
        except KeyError:
            raise ValueError("string strength must be 'required', 'strong', 'medium', "
                             f"or 'weak', not '{value}'") from None
    return _to_double(value)


def _to_double(value):
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Expected object of type `float, int, or long`. "
                    f"Got object of type `{type(value).__name__}` instead.")


def _number(value):
    """*value* as a float if kiwisolver does arithmetic with it, else None.

    kiwisolver accepts Python floats and ints and returns NotImplemented for
    other numbers (e.g. NumPy scalars), which then multiply through NumPy's
    object loop with the same result.
    """
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


_ids = itertools.count(1)


class _Symbolic:
    __slots__ = ()

    def _expr(self):
        raise NotImplementedError

    def __neg__(self):
        return self * -1.0

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(self, other, reverse=True)

    def __sub__(self, other):
        return _sub(self, other)

    def __rsub__(self, other):
        return _sub(self, other, reverse=True)

    def __truediv__(self, other):
        d = _number(other)
        if d is None or isinstance(other, _Symbolic):
            return NotImplemented
        if d == 0.0:
            raise ZeroDivisionError("float division by zero")
        return self * (1.0 / d)

    def __eq__(self, other):
        return _constraint(self, other, 2)

    def __le__(self, other):
        return _constraint(self, other, 0)

    def __ge__(self, other):
        return _constraint(self, other, 1)

    def __lt__(self, other):
        raise TypeError(f"unsupported operand type(s) for <: "
                        f"'{type(self).__name__}' and '{type(other).__name__}'")

    def __gt__(self, other):
        raise TypeError(f"unsupported operand type(s) for >: "
                        f"'{type(self).__name__}' and '{type(other).__name__}'")

    def __ne__(self, other):
        raise TypeError(f"unsupported operand type(s) for !=: "
                        f"'{type(self).__name__}' and '{type(other).__name__}'")

    __hash__ = None


class Variable(_Symbolic):
    """A variable of the solver; ``value()`` is set by `.Solver.updateVariables`."""

    __slots__ = ("_name", "_context", "_value", "_id", "__weakref__")

    def __init__(self, name="", context=None):
        if not isinstance(name, str):
            raise TypeError(f"Expected object of type `str`. "
                            f"Got object of type `{type(name).__name__}` instead.")
        self._name = name
        self._context = context
        self._value = 0.0
        self._id = next(_ids)

    def name(self):
        return self._name

    def setName(self, name):
        self._name = name

    def context(self):
        return self._context

    def setContext(self, context):
        self._context = context

    def value(self):
        return self._value

    def __repr__(self):
        return self._name

    def __mul__(self, other):
        c = _number(other)
        if c is None or isinstance(other, _Symbolic):
            return NotImplemented
        return Term(self, c)

    __rmul__ = __mul__

    def _expr(self):
        return Expression((Term(self, 1.0),))


class Term(_Symbolic):
    __slots__ = ("_variable", "_coefficient")

    def __init__(self, variable, coefficient=1.0):
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected object of type `Variable`. "
                            f"Got object of type `{type(variable).__name__}` instead.")
        self._variable = variable
        self._coefficient = _to_double(coefficient)

    def variable(self):
        return self._variable

    def coefficient(self):
        return self._coefficient

    def value(self):
        return self._coefficient * self._variable._value

    def __repr__(self):
        return f"{self._coefficient:g} * {self._variable._name}"

    def __mul__(self, other):
        c = _number(other)
        if c is None or isinstance(other, _Symbolic):
            return NotImplemented
        return Term(self._variable, self._coefficient * c)

    __rmul__ = __mul__


class Expression(_Symbolic):
    __slots__ = ("_terms", "_constant")

    def __init__(self, terms, constant=0.0):
        terms = tuple(terms)
        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"Expected object of type `Term`. "
                                f"Got object of type `{type(t).__name__}` instead.")
        self._terms = terms
        self._constant = _to_double(constant)

    def terms(self):
        return self._terms

    def constant(self):
        return self._constant

    def value(self):
        result = self._constant
        for t in self._terms:
            result += t._coefficient * t._variable._value
        return result

    def __repr__(self):
        return "".join(f"{t!r} + " for t in self._terms) + f"{self._constant:g}"

    def __mul__(self, other):
        c = _number(other)
        if c is None or isinstance(other, _Symbolic):
            return NotImplemented
        return Expression(tuple(t * c for t in self._terms), self._constant * c)

    __rmul__ = __mul__


def _term(x):
    return Term(x, 1.0) if isinstance(x, Variable) else x


def _add(first, second, reverse=False):
    # kiwisolver: Variable -> Term(v, 1.0); Term + Term -> Expression;
    # Expression + Term appends; numbers are added to the constant.
    if not isinstance(second, _Symbolic):
        c = _number(second)
        if c is None:
            return NotImplemented
        a = _term(first)
        if isinstance(a, Term):
            return Expression((a,), c)
        return Expression(a._terms, a._constant + c)
    a, b = (_term(second), _term(first)) if reverse else (_term(first), _term(second))
    if isinstance(a, Expression) and isinstance(b, Expression):
        return Expression(a._terms + b._terms, a._constant + b._constant)
    if isinstance(a, Expression):
        return Expression(a._terms + (b,), a._constant)
    if isinstance(b, Expression):
        return Expression(b._terms + (a,), b._constant)
    return Expression((a, b), 0.0)


def _sub(first, second, reverse=False):
    if reverse:
        # number - symbolic
        c = _number(second)
        if c is None or isinstance(second, _Symbolic):
            return NotImplemented
        return _add(-first, c)
    if not isinstance(second, _Symbolic):
        c = _number(second)
        if c is None:
            return NotImplemented
        return _add(first, -c)
    return _add(first, -second)


class Constraint:
    """A linear constraint ``expression <op> 0`` with a strength."""

    __slots__ = ("_expression", "_terms", "_op", "_strength", "_id", "__weakref__")

    def __init__(self, expression, op, strength="required"):
        if not isinstance(expression, Expression):
            raise TypeError(f"Expected object of type `Expression`. "
                            f"Got object of type `{type(expression).__name__}` instead.")
        if not isinstance(op, str):
            raise TypeError(f"Expected object of type `str`. "
                            f"Got object of type `{type(op).__name__}` instead.")
        if op not in _OPS:
            raise ValueError(f"relational operator must be '==', '<=', or '>=', not '{op}'")
        self._init(_reduce(expression), _OPS[op], _to_strength(strength))

    def _init(self, expression, op, strength):
        self._expression = expression
        self._op = op
        self._strength = _clip(strength)
        # kiwi reduces again (each variable occurs once here), ordered by
        # the address of each variable's data, i.e. by creation.
        self._terms = sorted(((t._variable._id, 0.0 + t._coefficient)
                              for t in expression._terms), key=lambda t: t[0])
        self._id = next(_ids)
        return self

    def expression(self):
        return self._expression

    def op(self):
        return {0: "<=", 1: ">=", 2: "=="}[self._op]

    def strength(self):
        return self._strength

    def violated(self):
        value = self._expression.value()
        if self._op == 2:
            return not (-value < 1.0e-8 if value < 0.0 else value < 1.0e-8)
        return value < 0.0 if self._op == 1 else value > 0.0

    def __or__(self, other):
        cn = Constraint.__new__(Constraint)
        cn._expression = self._expression
        cn._terms = self._terms
        cn._op = self._op
        cn._strength = _clip(_to_strength(other))
        cn._id = next(_ids)
        return cn

    __ror__ = __or__

    def __repr__(self):
        return (f"{self._expression!r} {self.op()} 0 | strength = {self._strength:g}"
                + (" (VIOLATED)" if self.violated() else ""))


def _reduce(expr):
    coeffs = {}
    variables = {}
    for t in expr._terms:
        key = t._variable._id
        coeffs[key] = coeffs.get(key, 0.0) + t._coefficient
        variables[key] = t._variable
    return Expression(tuple(Term(variables[k], c) for k, c in coeffs.items()),
                      expr._constant)


def _constraint(first, second, op):
    if not isinstance(second, _Symbolic) and _number(second) is None:
        return NotImplemented
    expr = _sub(first, second)
    if expr is NotImplemented:
        return expr
    return Constraint.__new__(Constraint)._init(_reduce(expr), op, strength.required)


class Solver:
    """kiwisolver's ``Solver``."""

    def __init__(self):
        self._impl = _mplrs.KiwiSolver()
        self._vars = {}  # id -> Variable, as kept alive by kiwi's solver
        self._cns = {}

    def _call(self, func, *args, obj=None):
        try:
            return func(*args)
        except ValueError as e:
            name = str(e)
            exc = {"DuplicateConstraint": DuplicateConstraint,
                   "UnsatisfiableConstraint": UnsatisfiableConstraint,
                   "UnknownConstraint": UnknownConstraint,
                   "DuplicateEditVariable": DuplicateEditVariable,
                   "UnknownEditVariable": UnknownEditVariable}.get(name)
            if exc is not None:
                raise exc(obj) from None
            if name == "BadRequiredStrength":
                raise BadRequiredStrength(
                    "A required strength cannot be used by an edit variable") from None
            raise RuntimeError(name) from None

    def addConstraint(self, constraint):
        if not isinstance(constraint, Constraint):
            raise TypeError(f"Expected object of type `Constraint`. "
                            f"Got object of type `{type(constraint).__name__}` instead.")
        for t in constraint._expression._terms:
            self._vars.setdefault(t._variable._id, t._variable)
        self._call(self._impl.add_constraint, constraint._id, constraint._terms,
                   constraint._expression._constant, constraint._op, constraint._strength,
                   obj=constraint)
        self._cns[constraint._id] = constraint

    def removeConstraint(self, constraint):
        self._call(self._impl.remove_constraint, constraint._id, obj=constraint)
        del self._cns[constraint._id]

    def hasConstraint(self, constraint):
        return self._impl.has_constraint(constraint._id)

    def addEditVariable(self, variable, strength):
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected object of type `Variable`. "
                            f"Got object of type `{type(variable).__name__}` instead.")
        self._vars.setdefault(variable._id, variable)
        self._call(self._impl.add_edit_variable, variable._id, _to_strength(strength),
                   obj=variable)

    def removeEditVariable(self, variable):
        self._call(self._impl.remove_edit_variable, variable._id, obj=variable)

    def hasEditVariable(self, variable):
        return self._impl.has_edit_variable(variable._id)

    def suggestValue(self, variable, value):
        self._call(self._impl.suggest_value, variable._id, _to_double(value), obj=variable)

    def updateVariables(self):
        for var_id, value in self._impl.values():
            self._vars[var_id]._value = value

    def reset(self):
        self._impl.reset()
        self._vars.clear()
        self._cns.clear()

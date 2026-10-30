//! Binding of the kiwi solver port, wrapped by `matplotlib_rs._kiwi`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::mpl::kiwi::{self, Constraint, Op, Solver, SolverError};

/// The solver core; `matplotlib_rs._kiwi.Solver` translates its error names
/// into kiwisolver's exceptions.
#[pyclass(name = "KiwiSolver", module = "matplotlib_rs._mplrs", unsendable)]
pub struct PyKiwiSolver {
    inner: Solver,
}

fn check(r: Result<(), SolverError>) -> PyResult<()> {
    r.map_err(|e| {
        PyValueError::new_err(match e {
            SolverError::DuplicateConstraint => "DuplicateConstraint",
            SolverError::UnsatisfiableConstraint => "UnsatisfiableConstraint",
            SolverError::UnknownConstraint => "UnknownConstraint",
            SolverError::DuplicateEditVariable => "DuplicateEditVariable",
            SolverError::UnknownEditVariable => "UnknownEditVariable",
            SolverError::BadRequiredStrength => "BadRequiredStrength",
            SolverError::Internal(msg) => msg,
        })
    })
}

#[pymethods]
impl PyKiwiSolver {
    #[new]
    fn new() -> Self {
        PyKiwiSolver { inner: Solver::new() }
    }

    /// Add constraint *key*: ``sum(c * var) + constant <op> 0`` with *op*
    /// 0 (<=), 1 (>=) or 2 (==) and a clipped *strength*.
    fn add_constraint(&mut self, key: u64, terms: Vec<(u64, f64)>, constant: f64, op: u8, strength: f64) -> PyResult<()> {
        let op = match op {
            0 => Op::Le,
            1 => Op::Ge,
            _ => Op::Eq,
        };
        check(self.inner.add_constraint(key, Constraint { terms, constant, op, strength: kiwi::clip(strength) }))
    }

    fn remove_constraint(&mut self, key: u64) -> PyResult<()> {
        check(self.inner.remove_constraint(key))
    }

    fn has_constraint(&self, key: u64) -> bool {
        self.inner.has_constraint(key)
    }

    fn add_edit_variable(&mut self, var: u64, strength: f64) -> PyResult<()> {
        check(self.inner.add_edit_variable(var, strength))
    }

    fn remove_edit_variable(&mut self, var: u64) -> PyResult<()> {
        check(self.inner.remove_edit_variable(var))
    }

    fn has_edit_variable(&self, var: u64) -> bool {
        self.inner.has_edit_variable(var)
    }

    fn suggest_value(&mut self, var: u64, value: f64) -> PyResult<()> {
        check(self.inner.suggest_value(var, value))
    }

    /// ``[(var, value)]`` for every variable known to the solver.
    fn values(&self) -> Vec<(u64, f64)> {
        self.inner.values()
    }

    fn reset(&mut self) {
        self.inner.reset()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKiwiSolver>()?;
    Ok(())
}

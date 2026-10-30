//! Port of the Cassowary solver of kiwi 1.5 (kiwisolver's C++ core, Modified
//! BSD license; see licenses/LICENSE_KIWISOLVER), used by constrained layout
//! when kiwisolver is not installed.
//!
//! The tableau, the pivoting rules and the order of every floating point
//! operation follow `kiwi/solverimpl.h` and `kiwi/row.h`, so that solutions
//! are identical.  kiwi orders the cells of a row (and its row map) by symbol
//! id, as here; the variables of a constraint come in the order chosen by the
//! caller (kiwisolver: the address of each variable, i.e. its creation order
//! in practice).

use std::collections::HashMap;

pub const REQUIRED: f64 = 1001001000.0;

/// `kiwi::strength::clip`.
pub fn clip(strength: f64) -> f64 {
    0.0f64.max(REQUIRED.min(strength))
}

fn near_zero(value: f64) -> bool {
    const EPS: f64 = 1.0e-8;
    if value < 0.0 {
        -value < EPS
    } else {
        value < EPS
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Invalid,
    External,
    Slack,
    Error,
    Dummy,
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    id: u64,
    kind: Kind,
}

impl Symbol {
    const INVALID: Symbol = Symbol { id: 0, kind: Kind::Invalid };
}

/// A sorted vector map (Loki's `AssocVector`, which kiwi uses).
#[derive(Clone, Default, Debug)]
struct Cells(Vec<(Symbol, f64)>);

impl Cells {
    fn pos(&self, id: u64) -> Result<usize, usize> {
        self.0.binary_search_by(|(s, _)| s.id.cmp(&id))
    }
    /// `m_cells[symbol] += value`, returning the new value.
    fn add(&mut self, symbol: Symbol, value: f64) -> f64 {
        match self.pos(symbol.id) {
            Ok(i) => {
                self.0[i].1 += value;
                self.0[i].1
            }
            Err(i) => {
                // operator[] inserts 0.0, to which the value is added
                let v = 0.0 + value;
                self.0.insert(i, (symbol, v));
                v
            }
        }
    }
    fn get(&self, id: u64) -> Option<f64> {
        self.pos(id).ok().map(|i| self.0[i].1)
    }
    fn erase(&mut self, id: u64) -> Option<f64> {
        self.pos(id).ok().map(|i| self.0.remove(i).1)
    }
}

#[derive(Clone, Default, Debug)]
struct Row {
    cells: Cells,
    constant: f64,
}

impl Row {
    fn new(constant: f64) -> Self {
        Row { cells: Cells::default(), constant }
    }
    fn add(&mut self, value: f64) -> f64 {
        self.constant += value;
        self.constant
    }
    fn insert_symbol(&mut self, symbol: Symbol, coefficient: f64) {
        if near_zero(self.cells.add(symbol, coefficient)) {
            self.cells.erase(symbol.id);
        }
    }
    fn insert_row(&mut self, other: &Row, coefficient: f64) {
        self.constant += other.constant * coefficient;
        for &(s, c) in &other.cells.0 {
            let coeff = c * coefficient;
            if near_zero(self.cells.add(s, coeff)) {
                self.cells.erase(s.id);
            }
        }
    }
    fn remove(&mut self, symbol: Symbol) {
        self.cells.erase(symbol.id);
    }
    fn reverse_sign(&mut self) {
        self.constant = -self.constant;
        for cell in &mut self.cells.0 {
            cell.1 = -cell.1;
        }
    }
    fn solve_for(&mut self, symbol: Symbol) {
        let coeff = -1.0 / self.cells.get(symbol.id).unwrap_or(0.0);
        self.cells.erase(symbol.id);
        self.constant *= coeff;
        for cell in &mut self.cells.0 {
            cell.1 *= coeff;
        }
    }
    fn solve_for_pair(&mut self, lhs: Symbol, rhs: Symbol) {
        self.insert_symbol(lhs, -1.0);
        self.solve_for(rhs);
    }
    fn coefficient_for(&self, symbol: Symbol) -> f64 {
        self.cells.get(symbol.id).unwrap_or(0.0)
    }
    fn substitute(&mut self, symbol: Symbol, row: &Row) {
        if let Some(coefficient) = self.cells.erase(symbol.id) {
            self.insert_row(row, coefficient);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Le,
    Ge,
    Eq,
}

/// A constraint: reduced terms `(variable, coefficient)`, constant, operator
/// and (clipped) strength.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub terms: Vec<(u64, f64)>,
    pub constant: f64,
    pub op: Op,
    pub strength: f64,
}

#[derive(Clone, Copy, Debug)]
struct Tag {
    marker: Symbol,
    other: Symbol,
}

struct EditInfo {
    tag: Tag,
    constraint: u64,
    constant: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SolverError {
    DuplicateConstraint,
    UnsatisfiableConstraint,
    UnknownConstraint,
    DuplicateEditVariable,
    UnknownEditVariable,
    BadRequiredStrength,
    Internal(&'static str),
}

/// A sorted map from symbol id to row (kiwi's `RowMap`).
#[derive(Default)]
struct Rows(Vec<(Symbol, Row)>);

impl Rows {
    fn pos(&self, id: u64) -> Result<usize, usize> {
        self.0.binary_search_by(|(s, _)| s.id.cmp(&id))
    }
    fn get(&self, s: Symbol) -> Option<&Row> {
        self.pos(s.id).ok().map(|i| &self.0[i].1)
    }
    fn get_mut(&mut self, s: Symbol) -> Option<&mut Row> {
        self.pos(s.id).ok().map(move |i| &mut self.0[i].1)
    }
    fn take(&mut self, s: Symbol) -> Option<Row> {
        self.pos(s.id).ok().map(|i| self.0.remove(i).1)
    }
    fn set(&mut self, s: Symbol, row: Row) {
        match self.pos(s.id) {
            Ok(i) => self.0[i].1 = row,
            Err(i) => self.0.insert(i, (s, row)),
        }
    }
}

#[derive(Default)]
pub struct Solver {
    cns: HashMap<u64, (Constraint, Tag)>,
    rows: Rows,
    vars: HashMap<u64, Symbol>,
    edits: HashMap<u64, EditInfo>,
    infeasible_rows: Vec<Symbol>,
    objective: Row,
    artificial: Option<Row>,
    id_tick: u64,
}

/// Constraint keys at and above this value belong to edit variables.
const EDIT_KEY: u64 = 1 << 63;

impl Solver {
    pub fn new() -> Self {
        Solver { id_tick: 1, ..Default::default() }
    }

    pub fn reset(&mut self) {
        *self = Solver::new();
    }

    pub fn has_constraint(&self, key: u64) -> bool {
        self.cns.contains_key(&key)
    }

    pub fn add_constraint(&mut self, key: u64, constraint: Constraint) -> Result<(), SolverError> {
        if self.cns.contains_key(&key) {
            return Err(SolverError::DuplicateConstraint);
        }
        let (mut row, tag) = self.create_row(&constraint);
        let mut subject = Self::choose_subject(&row, &tag);
        if subject.kind == Kind::Invalid && Self::all_dummies(&row) {
            if !near_zero(row.constant) {
                return Err(SolverError::UnsatisfiableConstraint);
            }
            subject = tag.marker;
        }
        if subject.kind == Kind::Invalid {
            if !self.add_with_artificial_variable(&row)? {
                return Err(SolverError::UnsatisfiableConstraint);
            }
        } else {
            row.solve_for(subject);
            self.substitute(subject, &row);
            self.rows.set(subject, row);
        }
        self.cns.insert(key, (constraint, tag));
        self.optimize(false)
    }

    pub fn remove_constraint(&mut self, key: u64) -> Result<(), SolverError> {
        let (constraint, tag) = self.cns.remove(&key).ok_or(SolverError::UnknownConstraint)?;
        self.remove_constraint_effects(&constraint, &tag);
        if self.rows.take(tag.marker).is_none() {
            let leaving = self
                .get_marker_leaving_row(tag.marker)
                .ok_or(SolverError::Internal("failed to find leaving row"))?;
            let mut row = self.rows.take(leaving).unwrap();
            row.solve_for_pair(leaving, tag.marker);
            self.substitute(tag.marker, &row);
        }
        self.optimize(false)
    }

    pub fn has_edit_variable(&self, var: u64) -> bool {
        self.edits.contains_key(&var)
    }

    pub fn add_edit_variable(&mut self, var: u64, strength: f64) -> Result<(), SolverError> {
        if self.edits.contains_key(&var) {
            return Err(SolverError::DuplicateEditVariable);
        }
        let strength = clip(strength);
        if strength == REQUIRED {
            return Err(SolverError::BadRequiredStrength);
        }
        let key = EDIT_KEY | var;
        let constraint = Constraint { terms: vec![(var, 1.0)], constant: 0.0, op: Op::Eq, strength };
        self.add_constraint(key, constraint)?;
        let tag = self.cns[&key].1;
        self.edits.insert(var, EditInfo { tag, constraint: key, constant: 0.0 });
        Ok(())
    }

    pub fn remove_edit_variable(&mut self, var: u64) -> Result<(), SolverError> {
        let key = self.edits.get(&var).ok_or(SolverError::UnknownEditVariable)?.constraint;
        self.remove_constraint(key)?;
        self.edits.remove(&var);
        Ok(())
    }

    pub fn suggest_value(&mut self, var: u64, value: f64) -> Result<(), SolverError> {
        let info = self.edits.get_mut(&var).ok_or(SolverError::UnknownEditVariable)?;
        let delta = value - info.constant;
        info.constant = value;
        let tag = info.tag;
        if let Some(row) = self.rows.get_mut(tag.marker) {
            if row.add(-delta) < 0.0 {
                self.infeasible_rows.push(tag.marker);
            }
        } else if let Some(row) = self.rows.get_mut(tag.other) {
            if row.add(delta) < 0.0 {
                self.infeasible_rows.push(tag.other);
            }
        } else {
            for (s, row) in &mut self.rows.0 {
                let coeff = row.coefficient_for(tag.marker);
                if coeff != 0.0 && row.add(delta * coeff) < 0.0 && s.kind != Kind::External {
                    self.infeasible_rows.push(*s);
                }
            }
        }
        self.dual_optimize()
    }

    /// The value of each variable known to the solver.
    pub fn values(&self) -> Vec<(u64, f64)> {
        self.vars
            .iter()
            .map(|(&var, &s)| (var, self.rows.get(s).map_or(0.0, |r| r.constant)))
            .collect()
    }

    fn var_symbol(&mut self, var: u64) -> Symbol {
        if let Some(&s) = self.vars.get(&var) {
            return s;
        }
        let s = Symbol { id: self.id_tick, kind: Kind::External };
        self.id_tick += 1;
        self.vars.insert(var, s);
        s
    }

    fn new_symbol(&mut self, kind: Kind) -> Symbol {
        let s = Symbol { id: self.id_tick, kind };
        self.id_tick += 1;
        s
    }

    fn create_row(&mut self, constraint: &Constraint) -> (Row, Tag) {
        let mut row = Row::new(constraint.constant);
        let mut tag = Tag { marker: Symbol::INVALID, other: Symbol::INVALID };
        for &(var, coefficient) in &constraint.terms {
            if !near_zero(coefficient) {
                let symbol = self.var_symbol(var);
                match self.rows.get(symbol) {
                    Some(basic) => row.insert_row(basic, coefficient),
                    None => row.insert_symbol(symbol, coefficient),
                }
            }
        }
        match constraint.op {
            Op::Le | Op::Ge => {
                let coeff = if constraint.op == Op::Le { 1.0 } else { -1.0 };
                let slack = self.new_symbol(Kind::Slack);
                tag.marker = slack;
                row.insert_symbol(slack, coeff);
                if constraint.strength < REQUIRED {
                    let error = self.new_symbol(Kind::Error);
                    tag.other = error;
                    row.insert_symbol(error, -coeff);
                    self.objective.insert_symbol(error, constraint.strength);
                }
            }
            Op::Eq => {
                if constraint.strength < REQUIRED {
                    let errplus = self.new_symbol(Kind::Error);
                    let errminus = self.new_symbol(Kind::Error);
                    tag.marker = errplus;
                    tag.other = errminus;
                    row.insert_symbol(errplus, -1.0);
                    row.insert_symbol(errminus, 1.0);
                    self.objective.insert_symbol(errplus, constraint.strength);
                    self.objective.insert_symbol(errminus, constraint.strength);
                } else {
                    let dummy = self.new_symbol(Kind::Dummy);
                    tag.marker = dummy;
                    row.insert_symbol(dummy, 1.0);
                }
            }
        }
        if row.constant < 0.0 {
            row.reverse_sign();
        }
        (row, tag)
    }

    fn choose_subject(row: &Row, tag: &Tag) -> Symbol {
        for &(s, _) in &row.cells.0 {
            if s.kind == Kind::External {
                return s;
            }
        }
        for s in [tag.marker, tag.other] {
            if (s.kind == Kind::Slack || s.kind == Kind::Error) && row.coefficient_for(s) < 0.0 {
                return s;
            }
        }
        Symbol::INVALID
    }

    fn add_with_artificial_variable(&mut self, row: &Row) -> Result<bool, SolverError> {
        let art = self.new_symbol(Kind::Slack);
        self.rows.set(art, row.clone());
        self.artificial = Some(row.clone());
        let result = self.optimize(true);
        let success = near_zero(self.artificial.take().unwrap().constant);
        result?;
        if let Some(mut r) = self.rows.take(art) {
            if r.cells.0.is_empty() {
                return Ok(success);
            }
            let entering = Self::any_pivotable_symbol(&r);
            if entering.kind == Kind::Invalid {
                return Ok(false);
            }
            r.solve_for_pair(art, entering);
            self.substitute(entering, &r);
            self.rows.set(entering, r);
        }
        for (_, r) in &mut self.rows.0 {
            r.remove(art);
        }
        self.objective.remove(art);
        Ok(success)
    }

    fn substitute(&mut self, symbol: Symbol, row: &Row) {
        for (s, r) in &mut self.rows.0 {
            r.substitute(symbol, row);
            if s.kind != Kind::External && r.constant < 0.0 {
                self.infeasible_rows.push(*s);
            }
        }
        self.objective.substitute(symbol, row);
        if let Some(a) = &mut self.artificial {
            a.substitute(symbol, row);
        }
    }

    /// Phase 2 of the simplex method for the objective function or, with
    /// `artificial`, the artificial objective (both are updated in place by
    /// `substitute` while pivoting).
    fn optimize(&mut self, artificial: bool) -> Result<(), SolverError> {
        loop {
            let obj = if artificial { self.artificial.as_ref().unwrap() } else { &self.objective };
            let entering = Self::entering_symbol(obj);
            if entering.kind == Kind::Invalid {
                return Ok(());
            }
            let leaving = self
                .leaving_row(entering)
                .ok_or(SolverError::Internal("The objective is unbounded."))?;
            let mut row = self.rows.take(leaving).unwrap();
            row.solve_for_pair(leaving, entering);
            self.substitute(entering, &row);
            self.rows.set(entering, row);
        }
    }

    fn dual_optimize(&mut self) -> Result<(), SolverError> {
        while let Some(leaving) = self.infeasible_rows.pop() {
            let entering = match self.rows.get(leaving) {
                Some(row) if !near_zero(row.constant) && row.constant < 0.0 => {
                    let entering = self.dual_entering_symbol(row);
                    if entering.kind == Kind::Invalid {
                        return Err(SolverError::Internal("Dual optimize failed."));
                    }
                    entering
                }
                _ => continue,
            };
            let mut row = self.rows.take(leaving).unwrap();
            row.solve_for_pair(leaving, entering);
            self.substitute(entering, &row);
            self.rows.set(entering, row);
        }
        Ok(())
    }

    fn entering_symbol(objective: &Row) -> Symbol {
        for &(s, c) in &objective.cells.0 {
            if s.kind != Kind::Dummy && c < 0.0 {
                return s;
            }
        }
        Symbol::INVALID
    }

    fn dual_entering_symbol(&self, row: &Row) -> Symbol {
        let mut entering = Symbol::INVALID;
        let mut ratio = f64::MAX;
        for &(s, c) in &row.cells.0 {
            if c > 0.0 && s.kind != Kind::Dummy {
                let coeff = self.objective.coefficient_for(s);
                let r = coeff / c;
                if r < ratio {
                    ratio = r;
                    entering = s;
                }
            }
        }
        entering
    }

    fn any_pivotable_symbol(row: &Row) -> Symbol {
        for &(s, _) in &row.cells.0 {
            if s.kind == Kind::Slack || s.kind == Kind::Error {
                return s;
            }
        }
        Symbol::INVALID
    }

    fn leaving_row(&self, entering: Symbol) -> Option<Symbol> {
        let mut ratio = f64::MAX;
        let mut found = None;
        for (s, r) in &self.rows.0 {
            if s.kind != Kind::External {
                let temp = r.coefficient_for(entering);
                if temp < 0.0 {
                    let temp_ratio = -r.constant / temp;
                    if temp_ratio < ratio {
                        ratio = temp_ratio;
                        found = Some(*s);
                    }
                }
            }
        }
        found
    }

    fn get_marker_leaving_row(&self, marker: Symbol) -> Option<Symbol> {
        let (mut r1, mut r2) = (f64::MAX, f64::MAX);
        let (mut first, mut second, mut third) = (None, None, None);
        for (s, row) in &self.rows.0 {
            let c = row.coefficient_for(marker);
            if c == 0.0 {
                continue;
            }
            if s.kind == Kind::External {
                third = Some(*s);
            } else if c < 0.0 {
                let r = -row.constant / c;
                if r < r1 {
                    r1 = r;
                    first = Some(*s);
                }
            } else {
                let r = row.constant / c;
                if r < r2 {
                    r2 = r;
                    second = Some(*s);
                }
            }
        }
        first.or(second).or(third)
    }

    fn remove_constraint_effects(&mut self, constraint: &Constraint, tag: &Tag) {
        if tag.marker.kind == Kind::Error {
            self.remove_marker_effects(tag.marker, constraint.strength);
        }
        if tag.other.kind == Kind::Error {
            self.remove_marker_effects(tag.other, constraint.strength);
        }
    }

    fn remove_marker_effects(&mut self, marker: Symbol, strength: f64) {
        match self.rows.get(marker) {
            Some(row) => {
                let row = row.clone();
                self.objective.insert_row(&row, -strength);
            }
            None => self.objective.insert_symbol(marker, -strength),
        }
    }

    fn all_dummies(row: &Row) -> bool {
        row.cells.0.iter().all(|(s, _)| s.kind == Kind::Dummy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cn(terms: &[(u64, f64)], constant: f64, op: Op, strength: f64) -> Constraint {
        Constraint { terms: terms.to_vec(), constant, op, strength: clip(strength) }
    }

    fn value(s: &Solver, var: u64) -> f64 {
        s.values().into_iter().find(|v| v.0 == var).unwrap().1
    }

    #[test]
    fn simple_system() {
        // x + y == 10 (required), x >= 2 (required), x == 8 (weak), y == 9 (strong)
        let mut s = Solver::new();
        s.add_constraint(1, cn(&[(1, 1.0), (2, 1.0)], -10.0, Op::Eq, REQUIRED)).unwrap();
        s.add_constraint(2, cn(&[(1, 1.0)], -2.0, Op::Ge, REQUIRED)).unwrap();
        s.add_constraint(3, cn(&[(1, 1.0)], -8.0, Op::Eq, 1.0)).unwrap();
        s.add_constraint(4, cn(&[(2, 1.0)], -9.0, Op::Eq, 1e6)).unwrap();
        assert_eq!(value(&s, 1), 2.0);
        assert_eq!(value(&s, 2), 8.0);
        assert_eq!(s.add_constraint(1, cn(&[(1, 1.0)], 0.0, Op::Eq, REQUIRED)), Err(SolverError::DuplicateConstraint));
        assert_eq!(
            s.add_constraint(5, cn(&[(1, 1.0)], 1.0, Op::Le, REQUIRED)),
            Err(SolverError::UnsatisfiableConstraint)
        );
        s.remove_constraint(2).unwrap();
        assert_eq!(value(&s, 2), 9.0);
        assert_eq!(value(&s, 1), 1.0);
    }

    #[test]
    fn edit_variables() {
        let mut s = Solver::new();
        s.add_edit_variable(1, 1e6).unwrap();
        assert_eq!(s.add_edit_variable(1, 1e6), Err(SolverError::DuplicateEditVariable));
        assert_eq!(s.add_edit_variable(2, REQUIRED), Err(SolverError::BadRequiredStrength));
        s.add_constraint(1, cn(&[(2, 1.0), (1, -2.0)], 0.0, Op::Eq, REQUIRED)).unwrap();
        s.suggest_value(1, 3.5).unwrap();
        assert_eq!(value(&s, 1), 3.5);
        assert_eq!(value(&s, 2), 7.0);
        s.remove_edit_variable(1).unwrap();
        assert_eq!(s.suggest_value(1, 1.0), Err(SolverError::UnknownEditVariable));
    }
}

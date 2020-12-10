use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod logic;

/// Wrappers around the core logic
#[pyfunction]
fn get_possible_actions(_stone: usize, _board: [[usize; 8]; 8]) -> PyResult<Vec<usize>> {
    Ok(logic::get_possible_actions(_stone, _board))
}
#[pyfunction]
pub fn place_tile(_x: usize, _y: usize, _color: usize, _board: [[usize; 8]; 8]) -> PyResult<Vec<[usize; 8]>> {
    Ok(logic::place_tile(_x, _y, _color, _board).to_vec())
}
#[pyfunction]
pub fn scores(_board: [[usize; 8]; 8]) -> PyResult<[usize; 2]> {
    Ok(logic::scores(_board))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_reversi(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_possible_actions, m)?)?;
    m.add_function(wrap_pyfunction!(place_tile, m)?)?;
    m.add_function(wrap_pyfunction!(scores, m)?)?;

    Ok(())
}

pub mod reed_solomon;
pub mod tensor_variant;

use ark_ff::fields::Field;
use ndarray::Array2 as Matrix;

use crate::error::Error;
pub struct DataGrid<A, B, T> {
    rows: A,
    cols: B,
    grid: Matrix<T>,
}

impl<A, B, T> DataGrid<A, B, T>
where
    A: Into<usize> + Copy,
    B: Into<usize> + Copy,
    T: Field,
{
    pub fn new(rows: A, cols: B, matrix: Matrix<T>) -> Result<Self, Error> {
        if matrix.len() != rows.into() {
            return Err(Error::LengthMismatch);
        }
        Ok(Self {
            rows,
            cols,
            grid: matrix,
        })
    }
}

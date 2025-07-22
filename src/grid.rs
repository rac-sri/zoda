use ark_ff::fields::Field;
use ndarray::Array2 as Matrix;

use crate::{error::Error, variants::Variant};

pub struct DataGrid<F, V> {
    pub grid: Matrix<F>,
    pub variant: V,
}

impl<F, V> DataGrid<F, V>
where
    F: Field,
    V: Variant<Matrix<F>>,
{
    pub fn new(grid: Matrix<F>, variant: V) -> Result<Self, Error> {
        Ok(Self { grid, variant })
    }

    pub fn encode(&mut self) -> Result<Matrix<F>, Error> {
        self.variant.encode(&self.grid)
    }
}

use std::marker::PhantomData;

use ark_ff::fields::Field;
use ndarray::Array2 as Matrix;

use crate::{error::Error, variants::Variant};

pub struct DataGrid<F, V, Q> {
    pub grid: Matrix<F>,
    pub variant: V,
    phantom: PhantomData<Q>,
}

impl<F, V, Q> DataGrid<F, V, Q>
where
    F: Field,
    V: Variant<Matrix<F>, Q>,
{
    pub fn new(grid: Matrix<F>, variant: V) -> Result<Self, Error> {
        Ok(Self {
            grid,
            variant,
            phantom: PhantomData,
        })
    }

    pub fn encode(&mut self) -> Result<Q, Error> {
        self.variant.encode(&self.grid)
    }
}

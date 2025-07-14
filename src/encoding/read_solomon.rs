use ark_ff::fields::Field;
use ndarray::Array2 as Matrix;
use reed_solomon_simd::{decode, encode};

macro_rules! create_matrix {
    ($entries:expr, $rows:expr, $cols:expr) => {
        Matrix::from_shape_vec(
            ($rows.into(), $cols.into()),
            $entries.into_iter().flatten().collect(),
        )
        .map_err(|e| Error::MatrixShapeError(e.to_string()))
    };
}

use crate::error::Error;
pub struct ReadSolomon {}

impl ReadSolomon {
    pub fn tensor_encode<T: Field, S: Into<usize> + Copy>(
        &self,
        matrix: &Matrix<T>,
        rows_size: S,
        cols_size: S,
    ) -> Result<Matrix<T>, Error> {
        let rows = self.encode_rows(matrix, rows_size, cols_size)?;
        let cols = self.encode_cols(&rows, rows_size, cols_size)?;
        Ok(cols)
    }

    pub fn encode_rows<T: Field, S: Into<usize> + Copy>(
        &self,
        matrix: &Matrix<T>,
        rows_size: S,
        cols_size: S,
    ) -> Result<Matrix<T>, Error> {
        let rows = matrix
            .rows()
            .into_iter()
            .map(|row| self.encode(&row.to_vec()).ok_or(Error::EncodingError))
            .collect::<Result<Vec<Vec<T>>, Error>>()?;

        create_matrix!(rows, rows_size, cols_size)
    }

    pub fn encode_cols<T: Field, S: Into<usize> + Copy>(
        &self,
        matrix: &Matrix<T>,
        rows_size: S,
        cols_size: S,
    ) -> Result<Matrix<T>, Error> {
        let cols = matrix
            .columns()
            .into_iter()
            .map(|col| self.encode(&col.to_vec()).ok_or(Error::EncodingError))
            .collect::<Result<Vec<Vec<T>>, Error>>()?;

        create_matrix!(cols, rows_size, cols_size)
    }

    fn encode<T: Field>(&self, items: &Vec<T>) -> Option<Vec<T>> {
        let bytes = items
            .iter()
            .map(|ele| {
                let mut buf = vec![];
                ele.serialize_uncompressed(&mut buf).unwrap(); // TODO: handle error
                buf
            })
            .collect::<Vec<Vec<u8>>>();

        let rs_encoding = encode(items.len(), items.len() * 2, bytes)
            .map_err(|e| Error::Custom(e.to_string()))
            .ok()?
            .iter()
            .map(|chunk| T::deserialize_uncompressed(&chunk[..]).unwrap()) // TODO: handle error
            .collect::<Vec<T>>();
        Some(rs_encoding)
    }

    fn decode<T: Field>(
        &self,
        original_shards: Vec<(usize, T)>,
        recovery_shards: Vec<(usize, T)>,
    ) -> Option<Vec<T>> {
        let original_bytes = original_shards
            .iter()
            .map(|ele| {
                let mut buf = vec![];
                ele.1.serialize_uncompressed(&mut buf).unwrap(); // TODO: handle error
                (ele.0, buf)
            })
            .collect::<Vec<(usize, Vec<u8>)>>();

        let recovery_bytes = recovery_shards
            .iter()
            .map(|ele| {
                let mut buf = vec![];
                ele.1.serialize_uncompressed(&mut buf).unwrap(); // TODO: handle error
                (ele.0, buf)
            })
            .collect::<Vec<(usize, Vec<u8>)>>();
        let rs_decoding = decode(
            original_shards.len(),
            recovery_shards.len() * 2,
            original_bytes,
            recovery_bytes,
        )
        .map_err(|e| Error::Custom(e.to_string()))
        .ok()?
        .into_iter()
        .map(|(_, chunk)| T::deserialize_uncompressed(&chunk[..]).unwrap()) // TODO: handle error
        .collect::<Vec<T>>();
        Some(rs_decoding)
    }
}

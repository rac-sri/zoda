use crate::error::Error;
use ark_ff::fields::Field;
use ndarray::{Array1, Array2 as Matrix, s};
use reed_solomon_simd::{algorithm, decode, encode, engine};

macro_rules! create_matrix {
    ($entries:expr, $rows:expr, $cols:expr) => {
        Matrix::from_shape_vec(
            ($rows.into(), $cols.into()),
            $entries.into_iter().flatten().collect(),
        )
        .map_err(|e| Error::MatrixShapeError(e.to_string()))
    };
}

pub struct ReedSolomon {
    reconstruction_factor: usize,
}

impl ReedSolomon {
    pub fn new(reconstruction_factor: usize) -> Self {
        Self {
            reconstruction_factor,
        }
    }
    pub fn tensor_encode<T: Field>(&self, matrix: &Matrix<T>) -> Result<Matrix<T>, Error> {
        let n = matrix.nrows();
        let mut matrix_tensor = Matrix::<T>::zeros((
            self.reconstruction_factor * n, // TODO: implementation assumed reconstruction factor as 2
            self.reconstruction_factor * n,
        ));

        matrix_tensor.slice_mut(s![0..n, 0..n]).assign(&matrix);

        let q2 = self.encode_rows(&matrix)?;

        matrix_tensor.slice_mut(s![0..n, n..2 * n]).assign(&q2);

        let q3 = self.encode_cols(&matrix)?;

        matrix_tensor.slice_mut(s![n..2 * n, 0..n]).assign(&q3);

        let q4 = self.encode_cols(&q2)?;

        matrix_tensor.slice_mut(s![n..2 * n, n..2 * n]).assign(&q4);

        Ok(matrix_tensor)
    }

    pub fn encode_rows<T: Field>(&self, matrix: &Matrix<T>) -> Result<Matrix<T>, Error> {
        let rows = matrix
            .rows()
            .into_iter()
            .map(|row| self.encode(&row.to_vec()).ok_or(Error::EncodingError))
            .collect::<Result<Vec<Vec<T>>, Error>>()?;

        create_matrix!(rows, matrix.nrows(), matrix.ncols())
    }

    pub fn encode_cols<T: Field>(&self, matrix: &Matrix<T>) -> Result<Matrix<T>, Error> {
        let cols = matrix
            .columns()
            .into_iter()
            .map(|col| self.encode(&col.to_vec()).ok_or(Error::EncodingError))
            .collect::<Result<Vec<Vec<T>>, Error>>()?;

        create_matrix!(cols, matrix.nrows(), matrix.ncols())
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

        let rs_encoding = encode(items.len(), items.len(), bytes)
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
            recovery_shards.len(),
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

    pub fn get_generator_matrixes<T: Field>(
        &self,
        rows: usize,
        cols: usize,
    ) -> Result<(Matrix<T>, Matrix<T>), Error> {
        let g_row = self.get_row_generator_matrix::<T>(cols)?;
        let g_col = self.get_col_generator_matrix::<T>(rows)?;
        Ok((g_row, g_col))
    }

    /// Extract the generator matrix G used for row encoding
    pub fn get_row_generator_matrix<T: Field>(&self, k: usize) -> Result<Matrix<T>, Error> {
        self.extract_generator_matrix_empirically(k)
    }

    /// Extract the generator matrix G^T used for column encoding  
    pub fn get_col_generator_matrix<T: Field>(&self, k: usize) -> Result<Matrix<T>, Error> {
        let g = self.extract_generator_matrix_empirically(k)?;
        Ok(g.t().to_owned())
    }

    pub fn extract_generator_matrix_empirically<T: Field>(
        &self,
        k: usize,
    ) -> Result<Matrix<T>, Error> {
        let mut matrix_rows = Vec::<Vec<T>>::new();

        for i in 0..k {
            let mut unit_vector = vec![T::zero(); k];
            unit_vector[i] = T::ONE;

            let encoded = self.encode(&unit_vector).ok_or(Error::EncodingError)?;

            matrix_rows.push(encoded);
        }

        println!("{:?} {:?} {:?}", matrix_rows.len(), matrix_rows[0].len(), k);
        let matrix = create_matrix!(matrix_rows, k, k)?;
        Ok(matrix)
    }
}

use crate::commitments;
use crate::encoding::Variant;
use crate::{
    commitments::ACCommitmentScheme, create_matrix, encoding::reed_solomon::ReedSolomon,
    error::Error,
};
use ark_ff::{FftField, Field};
use ndarray::{Array2 as Matrix, s};
use spongefish::{BytesToUnitSerialize, DefaultHash, DomainSeparator, UnitToBytes};
const TENSOR_VARIANT_DOMAIN_SEPARATOR: &str = "ZODA-TENSOR-VARIANT";
pub struct TensorVariant<F: Field, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    pub rs: ReedSolomon,
    generator: F,
    pub tensor_cache: Option<TensorVariantEncodingResult<F>>,
    pub commitment: C,
}

impl<F: Field, C> Variant<Matrix<F>> for TensorVariant<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    fn encode(&mut self, grid: &Matrix<F>) -> Result<Matrix<F>, Error> where {
        self.tensor_cache = Some(self.encode_vandermonde(grid)?);
        Ok(self.tensor_cache.as_ref().unwrap().z.clone())
    }
    fn decode(&mut self, original: &Matrix<F>, shards: &Matrix<F>) -> Result<Matrix<F>, Error> {
        !unimplemented!()
    }
}

impl<F: Field, C> TensorVariant<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    pub fn new(generator: F, commitment: C) -> Self {
        let rs = ReedSolomon {};
        Self {
            rs,
            generator,
            tensor_cache: None,
            commitment,
        }
    }

    #[allow(non_snake_case)]
    pub fn encode_vandermonde(
        &mut self,
        original_grid: &Matrix<F>,
    ) -> Result<TensorVariantEncodingResult<F>, Error> {
        let (z, G) = self.tensor_encode_vandermonde(&original_grid)?;

        let mut commitment_scheme = self.commitment.clone();
        let row_wise_commits = { self.row_wise_commit(&z, &mut commitment_scheme) };
        let col_wise_commits = { self.col_wise_commit(&z, &mut commitment_scheme) };

        if original_grid.nrows() != original_grid.ncols() {
            return Err(Error::MatrixDimsMismatch);
        }

        let tilde_g_r = self.diagnol_matrix_gen(original_grid.nrows(), original_grid.ncols())?;
        let tilde_g_r_2 = self.diagnol_matrix_gen(original_grid.ncols(), original_grid.ncols())?;

        let z_r = original_grid.dot(&G.t().to_owned()).dot(&tilde_g_r.1);
        let z_r_2 = original_grid.t().dot(&G.t()).dot(&tilde_g_r_2.1);

        Ok(TensorVariantEncodingResult {
            z,
            z_r,
            z_r_2,
            tilde_g_r,
            tilde_g_r_2,
            row_wise_commits,
            col_wise_commits,
        })
    }

    #[allow(non_snake_case)]
    pub fn encode_fft(
        &self,
        original_grid: &Matrix<F>,
    ) -> Result<TensorVariantEncodingResult<F>, Error>
    where
        C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>>,
        C::Commitment: std::convert::Into<Vec<u8>>,
        F: FftField,
    {
        let z = self.tensor_encode_fft(&original_grid)?;

        let mut commitment = self.commitment.clone();
        let row_wise_commits = self.row_wise_commit(&z, &mut commitment);
        let col_wise_commits = self.col_wise_commit(&z, &mut commitment);

        if original_grid.nrows() != original_grid.ncols() {
            return Err(Error::MatrixDimsMismatch);
        }

        let tilde_g_r = self.diagnol_matrix_gen(original_grid.nrows(), original_grid.ncols())?;
        let tilde_g_r_2 = self.diagnol_matrix_gen(original_grid.ncols(), original_grid.ncols())?;

        let n = original_grid.ncols();
        let z_r = z.slice(s![0..n, n..2 * n]).dot(&tilde_g_r.1); // XGgr  ( G = G')
        let z_r_2 = z.slice(s![n..2 * n, 0..n]).t().dot(&tilde_g_r_2.1); // X^T.G^T.g' = (GX)^T.g'

        Ok(TensorVariantEncodingResult {
            z,
            z_r,
            z_r_2,
            tilde_g_r,
            tilde_g_r_2,
            row_wise_commits,
            col_wise_commits,
        })
    }

    fn tensor_encode_vandermonde(
        &self,
        original_grid: &Matrix<F>,
    ) -> Result<(Matrix<F>, Matrix<F>), Error> {
        let n = original_grid.nrows();

        let mut matrix_tensor = Matrix::<F>::zeros((n * 2, n * 2));

        // calculate Z = GXG'
        matrix_tensor
            .slice_mut(s![0..n, 0..n])
            .assign(&original_grid);

        let alphas = self.rs.alphas_with_generator(n, self.generator);

        let vandermonte_matrix_g = self.rs.vandermonde_matrix(&alphas, n)?; // cache this as an optimisation

        let row_encoding = self.rs.rs_encode(&original_grid, &vandermonte_matrix_g)?;

        matrix_tensor
            .slice_mut(s![0..n, n..2 * n])
            .assign(&row_encoding);

        let col_encoding = self
            .rs
            .rs_encode(&vandermonte_matrix_g.t().to_owned(), &original_grid)?;

        matrix_tensor
            .slice_mut(s![n..2 * n, 0..n])
            .assign(&col_encoding);

        let q4 = self.rs.rs_encode(&col_encoding, &vandermonte_matrix_g)?;

        matrix_tensor.slice_mut(s![n..2 * n, n..2 * n]).assign(&q4);
        Ok((matrix_tensor, vandermonte_matrix_g))
    }

    fn tensor_encode_fft(&self, original_grid: &Matrix<F>) -> Result<Matrix<F>, Error>
    where
        F: FftField,
    {
        let n = original_grid.nrows();
        let mut matrix_tensor = Matrix::<F>::zeros((n * 2, n * 2));

        matrix_tensor
            .slice_mut(s![0..n, 0..n])
            .assign(&original_grid);

        // row wise commitment
        let encoded_rows: Vec<Vec<F>> = original_grid
            .rows()
            .into_iter()
            .map(|row| self.rs.rs_encode_fft(&row.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_rows_matrix =
            create_matrix!(encoded_rows, encoded_rows.len(), encoded_rows[0].len())?;

        matrix_tensor
            .slice_mut(s!(0..n, n..2 * n))
            .assign(&encoded_rows_matrix);

        // encode q1 column wise
        let encoded_cols: Vec<Vec<F>> = original_grid
            .columns()
            .into_iter()
            .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_cols_matrix =
            create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?;

        matrix_tensor
            .slice_mut(s!(n..2 * n, 0..n))
            .assign(&encoded_cols_matrix);

        // encode q2 column wise
        let encoded_q2_cols: Vec<Vec<F>> = encoded_rows_matrix
            .columns()
            .into_iter()
            .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_q2_cols_matrix = create_matrix!(
            encoded_q2_cols,
            encoded_q2_cols.len(),
            encoded_q2_cols[0].len()
        )?;

        matrix_tensor
            .slice_mut(s!(n..2 * n, n..2 * n))
            .assign(&encoded_q2_cols_matrix);

        Ok(matrix_tensor)
    }

    fn row_wise_commit(
        &self,
        tensor_encode_matrix: &Matrix<F>,
        commitment_scheme: &mut C,
    ) -> Vec<Vec<u8>>
    where
        C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>>,
        C::Commitment: std::convert::Into<Vec<u8>>,
    {
        let rows_vec: Vec<Vec<F>> = tensor_encode_matrix
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let commitment_vec = rows_vec
            .iter()
            .map(|item| {
                let mut serialised_list = Vec::<Vec<u8>>::new();
                for e in item {
                    let mut writer = Vec::new();
                    e.serialize_uncompressed(&mut writer).unwrap();
                    serialised_list.push(writer);
                }

                commitment_scheme.commit(&serialised_list).unwrap().into()
            })
            .collect();

        commitment_vec
    }

    fn col_wise_commit(
        &self,
        tensor_encode_matrix: &Matrix<F>,
        commitment_scheme: &mut C,
    ) -> Vec<Vec<u8>>
    where
        C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>>,
        C::Commitment: std::convert::Into<Vec<u8>>,
    {
        let cols_vec: Vec<Vec<F>> = tensor_encode_matrix
            .columns()
            .into_iter()
            .map(|col| col.to_vec())
            .collect();

        let commitment_vec = cols_vec
            .iter()
            .map(|item| {
                let mut serialised_list = Vec::<Vec<u8>>::new();
                for e in item {
                    let mut writer = Vec::new();
                    e.serialize_uncompressed(&mut writer).unwrap();
                    serialised_list.push(writer);
                }

                commitment_scheme.commit(&serialised_list).unwrap().into()
            })
            .collect::<Vec<Vec<u8>>>();

        commitment_vec
    }

    fn diagnol_matrix_gen(&self, rows: usize, cols: usize) -> Result<(Vec<u8>, Matrix<F>), Error> {
        if rows != cols {
            return Err(Error::MatrixShapeError(format!(
                "Rows {} : Cols {}",
                rows, cols
            )));
        }

        let mut diagonal_matrix = Matrix::<F>::zeros((rows, cols));
        let mut combined_transcript = Vec::new();

        for i in 0..rows {
            // Create a fresh domain separator for each element
            let domain_separator =
                DomainSeparator::<DefaultHash>::new(TENSOR_VARIANT_DOMAIN_SEPARATOR)
                    .absorb(32, "↪️")
                    .squeeze(16, "↩️");

            let mut prover_state = domain_separator.to_prover_state();

            diagonal_matrix[(i, i)] = F::rand(prover_state.rng());

            let mut dest = Vec::new();
            diagonal_matrix[(i, i)]
                .serialize_uncompressed(&mut dest)
                .unwrap(); // TODO: handle error

            // Absorb the bytes
            prover_state
                .add_bytes(&dest)
                .map_err(|e| Error::Custom(e.to_string()))?;

            // Generate challenge
            let mut chal = [0u8; 16];
            prover_state
                .fill_challenge_bytes(&mut chal)
                .map_err(|e| Error::Custom(e.to_string()))?;

            // Collect transcript
            combined_transcript.extend_from_slice(&prover_state.narg_string());
        }

        Ok((combined_transcript, diagonal_matrix))
    }
}

pub struct TensorVariantEncodingResult<F> {
    pub z: Matrix<F>,
    pub z_r: Matrix<F>,
    pub z_r_2: Matrix<F>,
    pub tilde_g_r: (Vec<u8>, Matrix<F>),
    pub tilde_g_r_2: (Vec<u8>, Matrix<F>),
    pub row_wise_commits: Vec<Vec<u8>>,
    pub col_wise_commits: Vec<Vec<u8>>,
}

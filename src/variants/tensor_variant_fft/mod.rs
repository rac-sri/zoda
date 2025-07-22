use crate::variants::Variant;
use crate::{
    commitments::ACCommitmentScheme, create_matrix, error::Error,
    variants::reed_solomon::ReedSolomon,
};
use ark_ff::FftField;
use ndarray::{Array2 as Matrix, s};
use spongefish::{BytesToUnitSerialize, DefaultHash, DomainSeparator, UnitToBytes};
const TENSOR_VARIANT_DOMAIN_SEPARATOR: &str = "ZODA-TENSOR-VARIANT";
pub struct TensorVariantFft<F: FftField, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    pub rs: ReedSolomon,
    generator: F,
    pub tensor_cache: Option<TensorVariantEncodingResult<F>>,
    pub commitment: C,
}

impl<F: FftField, C> Variant<Matrix<F>> for TensorVariantFft<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    fn encode(&mut self, grid: &Matrix<F>) -> Result<Matrix<F>, Error> where {
        self.tensor_cache = Some(self.encode_fft(grid)?);
        Ok(self.tensor_cache.as_ref().unwrap().z.clone())
    }
    fn decode(&mut self, original: &Matrix<F>, shards: &Matrix<F>) -> Result<Matrix<F>, Error> {
        !unimplemented!()
    }
}

impl<F: FftField, C> TensorVariantFft<F, C>
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

        let tilde_g_r = self.random_vec(z.ncols(), 1)?;
        let tilde_g_r_2 = self.random_vec(z.nrows(), 1)?;

        let encoded_rows: Vec<Vec<F>> = original_grid
            .rows()
            .into_iter()
            .map(|row| self.rs.rs_encode_fft(&row.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_rows_matrix =
            create_matrix!(encoded_rows, encoded_rows.len(), encoded_rows[0].len())?;

        let encoded_cols: Vec<Vec<F>> = original_grid
            .columns()
            .into_iter()
            .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_cols_matrix =
            create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?
                .t()
                .to_owned();

        let z_r = encoded_rows_matrix.dot(&tilde_g_r.1); // XGgr  ( G = G')

        let z_r_2 = encoded_cols_matrix.t().dot(&tilde_g_r_2.1); // X^T.G^T.g' = (GX)^T.g'

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

    fn tensor_encode_fft(&self, original_grid: &Matrix<F>) -> Result<Matrix<F>, Error>
    where
        F: FftField,
    {
        let encoded_rows: Vec<Vec<F>> = original_grid
            .rows()
            .into_iter()
            .map(|row| self.rs.rs_encode_fft(&row.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_rows_matrix =
            create_matrix!(encoded_rows, encoded_rows.len(), encoded_rows[0].len())?;

        let encoded_cols: Vec<Vec<F>> = encoded_rows_matrix
            .columns()
            .into_iter()
            .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
            .collect::<Result<_, _>>()?;

        let encoded_cols_matrix =
            create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?;

        Ok(encoded_cols_matrix.t().to_owned())
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

    fn random_vec(&self, rows: usize, cols: usize) -> Result<(Vec<u8>, Matrix<F>), Error> {
        let mut random_matrix = Matrix::<F>::zeros((rows, cols));
        let mut combined_transcript = Vec::new();

        for i in 0..rows {
            for j in 0..cols {
                // Create a fresh domain separator for each element
                let domain_separator =
                    DomainSeparator::<DefaultHash>::new(TENSOR_VARIANT_DOMAIN_SEPARATOR)
                        .absorb(32, "↪️")
                        .squeeze(16, "↩️");

                let mut prover_state = domain_separator.to_prover_state();

                random_matrix[(i, j)] = F::rand(prover_state.rng());

                let mut dest = Vec::new();
                random_matrix[(i, j)]
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
        }

        Ok((combined_transcript, random_matrix))
    }

    pub fn sample_fft(
        &self,
        z_r: &Matrix<F>,
        z_r_2: &Matrix<F>,
        w: &Matrix<F>,
        y: &Matrix<F>,
        g_r: &Matrix<F>,
        g_r_2: &Matrix<F>,
    ) -> Result<(), Error> {
        // Check 6: WSg¯r = GSzr
        let z_r_columns: Vec<Vec<F>> = z_r.columns().into_iter().map(|col| col.to_vec()).collect();
        let encoded_z_r_columns: Vec<Vec<F>> = z_r_columns
            .iter()
            .map(|col| self.rs.rs_encode_fft(col))
            .collect::<Result<_, _>>()?;
        let g_s_z_r = create_matrix!(
            encoded_z_r_columns,
            encoded_z_r_columns[0].len(),
            encoded_z_r_columns.len()
        )?;

        if !(w.dot(g_r) == g_s_z_r) {
            return Err(Error::Custom(
                "Check 6: WSg¯r = GSzr check failed".to_string(),
            ));
        }

        // Check 7: (Y^T)S'g¯'r' = G'S'z'r'
        let z_r_prime_columns: Vec<Vec<F>> = z_r_2
            .columns()
            .into_iter()
            .map(|col| col.to_vec())
            .collect();
        let encoded_z_r_prime_columns: Vec<Vec<F>> = z_r_prime_columns
            .iter()
            .map(|col| self.rs.rs_encode_fft(col))
            .collect::<Result<_, _>>()?;
        let g_prime_s_prime_z_r_prime = create_matrix!(
            encoded_z_r_prime_columns,
            encoded_z_r_prime_columns[0].len(),
            encoded_z_r_prime_columns.len()
        )?;

        if !(y.dot(g_r_2) == g_prime_s_prime_z_r_prime) {
            return Err(Error::Custom(
                "Check 7: (Y^T)S'g¯'r' = G'S'z'r' check failed".to_string(),
            ));
        }

        // Check 8: g¯'T r' G zr = g¯T r G' z'r'
        if !(g_r_2.t().dot(&g_s_z_r) == g_r.t().dot(&g_prime_s_prime_z_r_prime)) {
            return Err(Error::Custom(
                "Check 8: g¯'T r' G zr = g¯T r G' z'r' failed".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TensorVariantEncodingResult<F> {
    pub z: Matrix<F>,
    pub z_r: Matrix<F>,
    pub z_r_2: Matrix<F>,
    pub tilde_g_r: (Vec<u8>, Matrix<F>),
    pub tilde_g_r_2: (Vec<u8>, Matrix<F>),
    pub row_wise_commits: Vec<Vec<u8>>,
    pub col_wise_commits: Vec<Vec<u8>>,
}

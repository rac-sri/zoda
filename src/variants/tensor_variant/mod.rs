use crate::variants::Variant;
use crate::{
    commitments::ACCommitmentScheme, create_matrix, error::Error,
    variants::reed_solomon::ReedSolomon,
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
        let row_wise_commits = self.row_wise_commit(&z, &mut commitment_scheme);
        let col_wise_commits = self.col_wise_commit(&z, &mut commitment_scheme);

        if original_grid.nrows() != original_grid.ncols() {
            return Err(Error::MatrixDimsMismatch);
        }

        let tilde_g_r = self.random_vec(z.nrows(), 1)?;
        let tilde_g_r_2 = self.random_vec(z.nrows(), 1)?;

        let z_r = original_grid.dot(&G.to_owned()).dot(&tilde_g_r.1);

        let z_r_2 = G.t().dot(&original_grid.to_owned()).t().dot(&tilde_g_r_2.1);

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

    // #[allow(non_snake_case)]
    // pub fn encode_fft(
    //     &self,
    //     original_grid: &Matrix<F>,
    // ) -> Result<TensorVariantEncodingResult<F>, Error>
    // where
    //     C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>>,
    //     C::Commitment: std::convert::Into<Vec<u8>>,
    //     F: FftField,
    // {
    //     let z = self.tensor_encode_fft(&original_grid)?;

    //     let mut commitment = self.commitment.clone();
    //     let row_wise_commits = self.row_wise_commit(&z, &mut commitment);
    //     let col_wise_commits = self.col_wise_commit(&z, &mut commitment);

    //     if original_grid.nrows() != original_grid.ncols() {
    //         return Err(Error::MatrixDimsMismatch);
    //     }

    //     let tilde_g_r = self.random_vec(original_grid.nrows(), original_grid.ncols())?;
    //     let tilde_g_r_2 = self.random_vec(original_grid.ncols(), original_grid.ncols())?;

    //     let n = original_grid.ncols();
    //     let z_r = z.slice(s![.., 0..n]).dot(&tilde_g_r.1); // XGgr  ( G = G')

    //     let z_r_2 = z.slice(s![0..n, ..]).t().dot(&tilde_g_r_2.1); // X^T.G^T.g' = (GX)^T.g'

    //     Ok(TensorVariantEncodingResult {
    //         z,
    //         z_r,
    //         z_r_2,
    //         tilde_g_r,
    //         tilde_g_r_2,
    //         row_wise_commits,
    //         col_wise_commits,
    //     })
    // }

    fn tensor_encode_vandermonde(
        &self,
        original_grid: &Matrix<F>,
    ) -> Result<(Matrix<F>, Matrix<F>), Error> {
        let n = original_grid.nrows();

        let alphas = self.rs.alphas_with_generator(2 * n, self.generator);

        let g = self.rs.vandermonde_matrix(&alphas, n, 2 * n)?; // cache this as an optimisation

        let tilde_g = g.t().to_owned();

        let row_encoding = self.rs.rs_encode(&original_grid, &g)?;

        let col_encoding = self.rs.rs_encode(&tilde_g, &row_encoding)?;

        Ok((col_encoding, g))
    }

    // fn tensor_encode_fft(&self, original_grid: &Matrix<F>) -> Result<Matrix<F>, Error>
    // where
    //     F: FftField,
    // {
    //     let n = original_grid.nrows();
    //     let mut matrix_tensor = Matrix::<F>::zeros((n * 2, n * 2));

    //     matrix_tensor
    //         .slice_mut(s![0..n, 0..n])
    //         .assign(&original_grid);

    //     // row wise commitment
    //     let encoded_rows: Vec<Vec<F>> = original_grid
    //         .rows()
    //         .into_iter()
    //         .map(|row| self.rs.rs_encode_fft(&row.to_vec()))
    //         .collect::<Result<_, _>>()?;

    //     let encoded_rows_matrix =
    //         create_matrix!(encoded_rows, encoded_rows.len(), encoded_rows[0].len())?;

    //     matrix_tensor
    //         .slice_mut(s!(0..n, n..2 * n))
    //         .assign(&encoded_rows_matrix);

    //     // encode q1 column wise
    //     let encoded_cols: Vec<Vec<F>> = original_grid
    //         .columns()
    //         .into_iter()
    //         .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
    //         .collect::<Result<_, _>>()?;

    //     let encoded_cols_matrix =
    //         create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?;

    //     matrix_tensor
    //         .slice_mut(s!(n..2 * n, 0..n))
    //         .assign(&encoded_cols_matrix);

    //     // encode q2 column wise
    //     let encoded_q2_cols: Vec<Vec<F>> = encoded_rows_matrix
    //         .columns()
    //         .into_iter()
    //         .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
    //         .collect::<Result<_, _>>()?;

    //     let encoded_q2_cols_matrix = create_matrix!(
    //         encoded_q2_cols,
    //         encoded_q2_cols.len(),
    //         encoded_q2_cols[0].len()
    //     )?;

    //     matrix_tensor
    //         .slice_mut(s!(n..2 * n, n..2 * n))
    //         .assign(&encoded_q2_cols_matrix);

    //     Ok(matrix_tensor)
    // }

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

    pub fn sample_vandermonte(
        &self,
        n: usize,
        row_split_start: usize,
        row_split_end: usize,
        col_split_start: usize,
        col_split_end: usize,
        z_r: &Matrix<F>,
        z_r_2: &Matrix<F>,
        w: &Matrix<F>,
        y: &Matrix<F>,
        g_r: &Matrix<F>,
        g_r_2: &Matrix<F>,
    ) -> Result<(), Error> {
        if w.nrows() != row_split_end - row_split_start
            || y.nrows() != col_split_end - col_split_start
        {
            return Err(Error::LengthMismatch);
        }
        let alphas = self.rs.alphas_with_generator(2 * n, self.generator);
        let vandermonte_matrix_g = self.rs.vandermonde_matrix(&alphas, n, 2 * n)?; // cache this as an optimisation

        // 1. match and verify the commitments

        // 2. check W.g_r = G.z_r
        // let g_r = g_r.slice(s![row_split_start..row_split_end, ..]);

        if !(w.dot(g_r)
            == vandermonte_matrix_g
                .t()
                .slice(s![row_split_start..row_split_end, ..])
                .dot(&z_r.to_owned()))
        {
            return Err(Error::Custom(
                "Check 1: W.g_r = G.z_r check failed".to_string(),
            ));
        }

        if !(y.dot(g_r_2)
            == vandermonte_matrix_g
                .t()
                .slice(s![col_split_start..col_split_end, ..])
                .dot(&z_r_2.to_owned()))
        {
            return Err(Error::Custom(
                "Check 2: (Y^T).g'_r' = G'.z_r' check failed".to_string(),
            ));
        }

        if !(g_r_2
            .t()
            .dot(&vandermonte_matrix_g.t())
            .dot(&z_r.to_owned())
            == g_r
                .t()
                .dot(&vandermonte_matrix_g.t())
                .dot(&z_r_2.to_owned()))
        {
            return Err(Error::Custom(
                "Check 3: g'^T_r'.G.z_r = g_r^T.G'.z'_r' failed".to_string(),
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        commitments::merkle_commitment::{ACMerkleTree, LeafHash, TwoToOneHash},
        grid::DataGrid,
        variants,
    };
    use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
    use ark_ff::BigInt;
    use ark_serialize::CanonicalSerialize;
    use ndarray::{arr2, s};
    use rand::{Rng, thread_rng};

    use ark_bls12_381::Fr as Fq;

    fn generate_mock_grid() -> Matrix<Fq> {
        let matrix = arr2(&[
            [
                Fq::new(BigInt::from(11_u8)),
                Fq::new(BigInt::from(32_u8)),
                Fq::new(BigInt::from(3_u8)),
                Fq::new(BigInt::from(13_u8)),
            ],
            [
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(4_u8)),
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(3_u8)),
            ],
            [
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(3_u8)),
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(3_u8)),
            ],
            [
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(4_u8)),
                Fq::new(BigInt::from(1_u8)),
                Fq::new(BigInt::from(3_u8)),
            ],
        ]);

        matrix
    }

    #[test]
    fn test_tensor_variant_encoding_result() {
        let matrix = generate_mock_grid();
        let mut leaves: Vec<Vec<u8>> = Vec::new();
        for _ in 0..4 {
            let fq = Fq::new(BigInt::from(1_u8));
            let mut buf = Vec::new();
            fq.serialize_uncompressed(&mut buf).unwrap();
            leaves.push(buf);
        }

        let mut rng = thread_rng();
        let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
        let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

        let ac_commit = ACMerkleTree::new(leaf_crh_params, two_to_one_crh_params, leaves).unwrap();
        let tensor_obj = variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit);

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        original_grid.encode().unwrap();

        let vals = original_grid.variant.tensor_cache.as_ref().unwrap();

        let n = original_grid.grid.nrows(); // assuming square matrix
        let num_samples = 20; // how many random samples you want

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Pick random start indices for rows and columns
            let row_split_start = rng.gen_range(0..(n - 1));
            let row_split_end = row_split_start + 2;
            let col_split_start = rng.gen_range(0..(n - 1));
            let col_split_end = col_split_start + 2;

            let sample_w = vals
                .z
                .slice(s![row_split_start..row_split_end, ..])
                .to_owned();

            let sample_y = vals
                .z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &vals.z_r,
                &vals.z_r_2,
                &sample_w,
                &sample_y,
                &vals.tilde_g_r.1,
                &vals.tilde_g_r_2.1,
            );

            assert!(result.is_ok(), "Random sample failed: {:?}", result);
        }
    }

    #[test]
    fn test_tensor_variant_sampling_failure_on_incorrect_data() {
        let matrix = generate_mock_grid();
        let mut leaves: Vec<Vec<u8>> = Vec::new();
        for _ in 0..4 {
            let fq = Fq::new(BigInt::from(1_u8));
            let mut buf = Vec::new();
            fq.serialize_uncompressed(&mut buf).unwrap();
            leaves.push(buf);
        }

        let mut rng = thread_rng();
        let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
        let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

        let ac_commit = ACMerkleTree::new(leaf_crh_params, two_to_one_crh_params, leaves).unwrap();
        let tensor_obj = variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit);

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        original_grid.encode().unwrap();

        // Mutate the encoded data
        let vals = original_grid.variant.tensor_cache.as_mut().unwrap();
        vals.z[[0, 0]] += Fq::from(1u64); // Corrupt the first element

        // Copy the data you need
        let z = vals.z.clone();
        let z_r = vals.z_r.clone();
        let z_r_2 = vals.z_r_2.clone();
        let tilde_g_r = vals.tilde_g_r.1.clone();
        let tilde_g_r_2 = vals.tilde_g_r_2.1.clone();

        // Now the mutable borrow is dropped, you can call methods on original_grid.variant
        let n = original_grid.grid.nrows();
        let num_samples = 100;
        let mut rng = rand::thread_rng();
        let mut any_failed = false;

        for _ in 0..num_samples {
            let row_split_start = rng.gen_range(0..(n - 1));
            let row_split_end = row_split_start + 2;
            let col_split_start = rng.gen_range(0..(n - 1));
            let col_split_end = col_split_start + 2;

            let sample_w = z.slice(s![row_split_start..row_split_end, ..]).to_owned();
            let sample_y = z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &z_r,
                &z_r_2,
                &sample_w,
                &sample_y,
                &tilde_g_r,
                &tilde_g_r_2,
            );

            if result.is_err() {
                any_failed = true;
            }
        }

        assert!(any_failed, "Sampling did not fail on corrupted data!");
    }
}

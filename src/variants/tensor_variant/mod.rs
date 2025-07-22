use crate::variants::Variant;
use crate::{commitments::ACCommitmentScheme, error::Error, variants::reed_solomon::ReedSolomon};
use ark_ff::Field;
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
        let (z, G, G_2) = self.tensor_encode_vandermonde(&original_grid)?;

        let mut commitment_scheme = self.commitment.clone();
        let row_wise_commits = self.row_wise_commit(&z, &mut commitment_scheme);
        let col_wise_commits = self.col_wise_commit(&z, &mut commitment_scheme);

        let tilde_g_r = self.random_vec(z.ncols(), 1)?;
        let tilde_g_r_2 = self.random_vec(z.nrows(), 1)?;

        let z_r = original_grid.dot(&G_2.t().to_owned()).dot(&tilde_g_r.1);
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

    fn tensor_encode_vandermonde(
        &self,
        original_grid: &Matrix<F>,
    ) -> Result<(Matrix<F>, Matrix<F>, Matrix<F>), Error> {
        let n = original_grid.nrows();
        let n_2 = original_grid.ncols();

        let alphas = self.rs.alphas_with_generator(2 * n, self.generator);
        let alphas_2 = self.rs.alphas_with_generator(2 * n_2, self.generator);

        let g = self.rs.vandermonde_matrix(&alphas, n, 2 * n)?; // cache this as an optimisation
        let g_2 = self.rs.vandermonde_matrix(&alphas_2, n_2, 2 * n_2)?;

        let row_encoding = self.rs.rs_encode(&g, &original_grid)?;

        let col_encoding = self.rs.rs_encode(&row_encoding, &g_2.t().to_owned())?;

        Ok((col_encoding, g, g_2))
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

    pub fn sample_vandermonte(
        &self,
        n: usize,
        n_2: usize,
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
        let alphas_2 = self.rs.alphas_with_generator(2 * n_2, self.generator);

        let g = self
            .rs
            .vandermonde_matrix(&alphas, n, 2 * n)?
            .t()
            .to_owned(); // cache this as an optimisation
        let g_2 = self
            .rs
            .vandermonde_matrix(&alphas_2, n_2, 2 * n_2)?
            .t()
            .to_owned();

        // 1. match and verify the commitments

        // 2. check W.g_r = G.z_r
        // let g_r = g_r.slice(s![row_split_start..row_split_end, ..]);

        if !(w.dot(g_r)
            == g.t()
                .slice(s![row_split_start..row_split_end, ..])
                .dot(&z_r.to_owned()))
        {
            return Err(Error::Custom(
                "Check 1: W.g_r = G.z_r check failed".to_string(),
            ));
        }

        if !(y.dot(g_r_2)
            == g_2
                .t()
                .slice(s![col_split_start..col_split_end, ..])
                .dot(&z_r_2.to_owned()))
        {
            return Err(Error::Custom(
                "Check 2: (Y^T).g'_r' = G'.z_r' check failed".to_string(),
            ));
        }

        if !(g_r_2.t().dot(&g.t()).dot(&z_r.to_owned())
            == g_r.t().dot(&g_2.t()).dot(&z_r_2.to_owned()))
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
        create_matrix,
        grid::DataGrid,
        variants,
    };
    use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
    use ark_ff::{BigInt, FftField};
    use ark_serialize::CanonicalSerialize;
    use ndarray::s;
    use rand::{Rng, thread_rng};

    use ark_bls12_381::Fr as Fq;

    fn generate_mock_grid(rows: usize, cols: usize) -> Matrix<Fq> {
        let mut rng = thread_rng();
        let mut data: Vec<Vec<Fq>> = Vec::with_capacity(rows);

        for _ in 0..rows {
            let mut row_vec: Vec<Fq> = Vec::with_capacity(cols);
            for _ in 0..cols {
                let val = rng.gen_range(0..256); // or choose a suitable range for Fq
                row_vec.push(Fq::new(BigInt::from(val as u32)));
            }
            data.push(row_vec);
        }

        let matrix = create_matrix!(data, rows, cols);
        matrix.unwrap()
    }
    #[test]
    fn test_tensor_variant_square_encoding_result() {
        let n: usize = 4; // Number of rows
        let m: usize = 4; // Number of columns
        let matrix = generate_mock_grid(n, m);
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

        let num_samples = 20; // Number of random samples

        assert!(n >= 2 && m >= 2, "Matrix must be at least 2x2 for sampling");

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Pick random start index for a 2-row block
            let row_split_start = rng.gen_range(0..=(n - 2));
            let row_split_end = row_split_start + 4;

            // Pick random start index for a 2-column block
            let col_split_start = rng.gen_range(0..=(m - 2));
            let col_split_end = col_split_start + 2;

            // Extract 2xAll columns row block
            let sample_w = vals
                .z
                .slice(s![row_split_start..row_split_end, ..])
                .to_owned();

            // Extract All rows x 2 columns column block, then transpose if needed
            let sample_y = vals
                .z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                m,
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
    fn test_tensor_variant_row_rec_encoding_result() {
        let n: usize = 8; // Number of rows
        let m: usize = 4; // Number of columns
        let matrix = generate_mock_grid(n, m);
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

        let num_samples = 20; // Number of random samples

        assert!(n >= 2 && m >= 2, "Matrix must be at least 2x2 for sampling");

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Pick random start index for a 2-row block
            let row_split_start = rng.gen_range(0..=(n - 2));
            let row_split_end = row_split_start + 2;

            // Pick random start index for a 2-column block
            let col_split_start = rng.gen_range(0..=(m - 2));
            let col_split_end = col_split_start + 2;

            // Extract 2xAll columns row block
            let sample_w = vals
                .z
                .slice(s![row_split_start..row_split_end, ..])
                .to_owned();

            // Extract All rows x 2 columns column block, then transpose if needed
            let sample_y = vals
                .z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                m,
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
    fn test_tensor_variant_column_rec_encoding_result() {
        let n: usize = 4; // Number of rows
        let m: usize = 8; // Number of columns
        let matrix = generate_mock_grid(n, m);
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

        let num_samples = 20; // Number of random samples

        assert!(n >= 2 && m >= 2, "Matrix must be at least 2x2 for sampling");

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Pick random start index for a 2-row block
            let row_split_start = rng.gen_range(0..=(n - 2));
            let row_split_end = row_split_start + 2;

            // Pick random start index for a 2-column block
            let col_split_start = rng.gen_range(0..=(m - 2));
            let col_split_end = col_split_start + 2;

            // Extract 2xAll columns row block
            let sample_w = vals
                .z
                .slice(s![row_split_start..row_split_end, ..])
                .to_owned();

            // Extract All rows x 2 columns column block, then transpose if needed
            let sample_y = vals
                .z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                m,
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
        let n: usize = 4; // Number of rows
        let m: usize = 4; // Number of columns
        let matrix = generate_mock_grid(4, 4);
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

        let (z, z_r, z_r_2, tilde_g_r, tilde_g_r_2); // Declare before the scope

        {
            let vals = original_grid.variant.tensor_cache.as_mut().unwrap();
            vals.z[[0, 0]] += Fq::from(1u64); // Corrupt

            z = vals.z.clone();
            z_r = vals.z_r.clone();
            z_r_2 = vals.z_r_2.clone();
            tilde_g_r = vals.tilde_g_r.1.clone();
            tilde_g_r_2 = vals.tilde_g_r_2.1.clone();
        }

        let num_samples = 20; // Number of random samples
        let mut any_failed = false;

        assert!(n >= 2 && m >= 2, "Matrix must be at least 2x2 for sampling");

        let mut rng = rand::thread_rng();

        for _ in 0..num_samples {
            // Pick random start index for a 2-row block
            let row_split_start = rng.gen_range(0..=(n - 2));
            let row_split_end = row_split_start + 2;

            // Pick random start index for a 2-column block
            let col_split_start = rng.gen_range(0..=(m - 2));
            let col_split_end = col_split_start + 2;

            let sample_w = z.slice(s![row_split_start..row_split_end, ..]).to_owned();

            let sample_y = z
                .slice(s![.., col_split_start..col_split_end])
                .t()
                .to_owned();

            let result = original_grid.variant.sample_vandermonte(
                n,
                m,
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

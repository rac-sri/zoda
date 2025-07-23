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
    pub commitment: C,
    pub g: Matrix<F>,
    pub g_2: Matrix<F>,
    pub g_r: Matrix<F>,
    pub g_r_2: Matrix<F>,
}

impl<F: Field, C> Variant<Matrix<F>, TensorVariantEncodingResult<F>> for TensorVariant<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    fn encode(&mut self, grid: &Matrix<F>) -> Result<TensorVariantEncodingResult<F>, Error> {
        self.encode_vandermonde(grid)
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
    pub fn new(generator: F, commitment: C, n: usize, n_2: usize) -> Result<Self, Error> {
        let rs = ReedSolomon {};
        let alphas = rs.alphas_with_generator(2 * n, generator);
        let alphas_2 = rs.alphas_with_generator(2 * n_2, generator);

        let g = rs.vandermonde_matrix(&alphas, n, 2 * n)?;
        let g_2 = rs.vandermonde_matrix(&alphas_2, n_2, 2 * n_2)?;

        let g_r = Self::random_vec(2 * n_2, 1)?;
        let g_r_2 = Self::random_vec(2 * n, 1)?;

        Ok(Self {
            rs,
            commitment,
            g,
            g_2,
            g_r,
            g_r_2,
        })
    }

    #[allow(non_snake_case)]
    pub fn encode_vandermonde(
        &mut self,
        original_grid: &Matrix<F>,
    ) -> Result<TensorVariantEncodingResult<F>, Error> {
        let row_encoding = self.rs.rs_encode(&self.g, &original_grid)?;

        let col_encoding = self
            .rs
            .rs_encode(&original_grid, &self.g_2.t().to_owned())?;

        let q4 = self.rs.rs_encode(
            &row_encoding
                .slice(s![original_grid.nrows()..2 * original_grid.nrows(), ..])
                .to_owned(),
            &self
                .g_2
                .t()
                .slice(s![.., original_grid.ncols()..2 * original_grid.ncols()])
                .to_owned()
                .to_owned(),
        );

        let z = {
            let mut full_matrix =
                Matrix::<F>::zeros((2 * original_grid.nrows(), 2 * original_grid.ncols()));

            // q1: top-left (from row_encoding or col_encoding)
            full_matrix
                .slice_mut(s![..original_grid.nrows(), ..original_grid.ncols()])
                .assign(original_grid);

            // q2: top-right (from col_encoding)
            full_matrix
                .slice_mut(s![..original_grid.nrows(), original_grid.ncols()..])
                .assign(&col_encoding.slice(s![..original_grid.nrows(), original_grid.ncols()..]));

            // q3: bottom-left (from row_encoding)
            full_matrix
                .slice_mut(s![original_grid.nrows().., ..original_grid.ncols()])
                .assign(&row_encoding.slice(s![original_grid.nrows().., ..original_grid.ncols()]));

            // q4: bottom-right (from q4 calculation)
            full_matrix
                .slice_mut(s![original_grid.nrows().., original_grid.ncols()..])
                .assign(&q4?);

            full_matrix
        };

        let mut commitment_scheme = self.commitment.clone();
        let row_wise_commits = self.row_wise_commit(&z, &mut commitment_scheme);
        let col_wise_commits = self.col_wise_commit(&z, &mut commitment_scheme);

        let z_r = col_encoding.dot(&self.g_r);
        let z_r_2 = row_encoding.t().dot(&self.g_r_2);

        Ok(TensorVariantEncodingResult {
            z,
            z_r,
            z_r_2,
            row_wise_commits,
            col_wise_commits,
        })
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

    fn random_vec(rows: usize, cols: usize) -> Result<Matrix<F>, Error> {
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

        Ok(random_matrix)
    }

    pub fn sample_vandermonte(
        &self,
        row_split_start: usize,
        row_split_end: usize,
        col_split_start: usize,
        col_split_end: usize,
        z_r: &Matrix<F>,
        z_r_2: &Matrix<F>,
        w: &Matrix<F>,
        y: &Matrix<F>,
    ) -> Result<(), Error> {
        if w.nrows() != row_split_end - row_split_start
            || y.nrows() != col_split_end - col_split_start
        {
            return Err(Error::LengthMismatch);
        }

        // 1. match and verify the commitments

        // 2. check W.g_r = G.z_r
        // let g_r = g_r.slice(s![row_split_start..row_split_end, ..]);

        if !(w.dot(&self.g_r)
            == self
                .g
                .slice(s![row_split_start..row_split_end, ..])
                .dot(&z_r.to_owned()))
        {
            return Err(Error::Custom(
                "Check 1: W.g_r = G.z_r check failed".to_string(),
            ));
        }

        if !(y.dot(&self.g_r_2)
            == self
                .g_2
                .slice(s![col_split_start..col_split_end, ..])
                .dot(&z_r_2.to_owned()))
        {
            return Err(Error::Custom(
                "Check 2: (Y^T).g'_r' = G'.z_r' check failed".to_string(),
            ));
        }

        if !(self.g_r_2.t().dot(&self.g).dot(&z_r.to_owned())
            == self.g_r.t().dot(&self.g_2).dot(&z_r_2.to_owned()))
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
        let tensor_obj =
            variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit, n, m).unwrap();

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        let vals = original_grid.encode().unwrap();

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
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &vals.z_r,
                &vals.z_r_2,
                &sample_w,
                &sample_y,
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
        let tensor_obj =
            variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit, n, m).unwrap();

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        let vals = original_grid.encode().unwrap();

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
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &vals.z_r,
                &vals.z_r_2,
                &sample_w,
                &sample_y,
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
        let tensor_obj =
            variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit, n, m).unwrap();

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        let vals = original_grid.encode().unwrap();

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
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &vals.z_r,
                &vals.z_r_2,
                &sample_w,
                &sample_y,
            );

            assert!(result.is_ok(), "Random sample failed: {:?}", result);
        }
    }

    #[test]
    fn test_tensor_variant_sampling_failure_on_incorrect_data() {
        let n: usize = 4_usize; // Number of rows
        let m: usize = 4_usize; // Number of columns
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
        let tensor_obj =
            variants::tensor_variant::TensorVariant::new(Fq::GENERATOR, ac_commit, n, m).unwrap();

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        let mut vals = original_grid.encode().unwrap();

        let (z, z_r, z_r_2); // Declare before the scope

        {
            vals.z[[0, 0]] += Fq::from(1u64); // Corrupt

            z = vals.z.clone();
            z_r = vals.z_r.clone();
            z_r_2 = vals.z_r_2.clone();
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
                row_split_start,
                row_split_end,
                col_split_start,
                col_split_end,
                &z_r,
                &z_r_2,
                &sample_w,
                &sample_y,
            );

            if result.is_err() {
                any_failed = true;
            }
        }

        assert!(any_failed, "Sampling did not fail on corrupted data!");
    }
}

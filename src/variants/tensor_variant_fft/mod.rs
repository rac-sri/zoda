use std::sync::Arc;

use crate::variants::Variant;
use crate::{
    commitments::ACCommitmentScheme, create_matrix, error::Error,
    variants::reed_solomon::ReedSolomon,
};
use ark_ff::FftField;
use ndarray::{Array2 as Matrix, arr1, s};
use spongefish::{BytesToUnitSerialize, DefaultHash, DomainSeparator, UnitToBytes};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const TENSOR_VARIANT_DOMAIN_SEPARATOR: &str = "ZODA-TENSOR-VARIANT";
pub struct TensorVariantFft<F: FftField, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone + Send + Sync,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    pub rs: Arc<ReedSolomon>,
    pub commitment: C,
    pub g_r: Matrix<F>,
    pub g_r_2: Matrix<F>,
}

impl<F: FftField, C> Variant<Matrix<F>, TensorVariantEncodingResult<F>> for TensorVariantFft<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone + Send + Sync,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    fn encode(&mut self, grid: &Matrix<F>) -> Result<TensorVariantEncodingResult<F>, Error> {
        self.encode_fft(grid)
    }
    fn decode(&mut self, original: &Matrix<F>, shards: &Matrix<F>) -> Result<Matrix<F>, Error> {
        !unimplemented!()
    }
}

impl<F: FftField, C> TensorVariantFft<F, C>
where
    C: ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> + Clone + Send + Sync,
    C::Commitment: std::convert::Into<Vec<u8>>,
{
    pub fn new(commitment: C, n: usize, n_2: usize) -> Result<Self, Error> {
        if !n.is_power_of_two() {
            return Err(Error::Custom(format!("n must be a power of 2, got {}", n)));
        }
        if !n_2.is_power_of_two() {
            return Err(Error::Custom(format!(
                "n_2 must be a power of 2, got {}",
                n_2
            )));
        }
        let rs = ReedSolomon {};
        let g_r = Self::random_vec(2 * n_2, 1)?;
        let g_r_2 = Self::random_vec(2 * n, 1)?;
        let g_r_matrix = arr1(&g_r).insert_axis(ndarray::Axis(1));
        let g_r_2_matrix = arr1(&g_r_2).insert_axis(ndarray::Axis(1));
        Ok(Self {
            rs: Arc::new(rs),
            commitment,
            g_r: g_r_matrix,
            g_r_2: g_r_2_matrix,
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
        #[cfg(feature = "parallel")]
        let rs = self.rs.clone();
        let encoded_rows: Vec<Vec<F>> = {
            let rows: Vec<_> = original_grid.rows().into_iter().collect();

            #[cfg(feature = "parallel")]
            let encoded_rows = rows
                .par_iter()
                .map(|row| rs.rs_encode_fft(&row.to_vec()))
                .collect::<Result<_, _>>()?;

            #[cfg(not(feature = "parallel"))]
            let encoded_rows = rows
                .iter()
                .map(|row| self.rs.rs_encode_fft(&row.to_vec()))
                .collect::<Result<_, _>>()?;

            encoded_rows
        };

        let encoded_rows_matrix =
            create_matrix!(encoded_rows, encoded_rows.len(), encoded_rows[0].len())?;

        let encoded_cols: Vec<Vec<F>> = {
            let cols: Vec<_> = encoded_rows_matrix.columns().into_iter().collect();

            #[cfg(feature = "parallel")]
            let encoded_cols = cols
                .par_iter()
                .map(|col| rs.rs_encode_fft(&col.to_vec()))
                .collect::<Result<_, _>>()?;

            #[cfg(not(feature = "parallel"))]
            let encoded_cols = cols
                .iter()
                .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
                .collect::<Result<_, _>>()?;

            encoded_cols
        };

        let encoded_cols_matrix =
            create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?;

        let z = encoded_cols_matrix.t().to_owned();

        let mut commitment = self.commitment.clone();

        let row_wise_commits = self.row_wise_commit(&z, &mut commitment);
        let col_wise_commits = self.col_wise_commit(&z, &mut commitment);

        let z_r = encoded_rows_matrix.dot(&self.g_r); // XGgr  ( G = G')

        let encoded_cols: Vec<Vec<F>> = {
            let cols: Vec<_> = original_grid.columns().into_iter().collect();

            #[cfg(feature = "parallel")]
            let encoded_cols = cols
                .par_iter()
                .map(|col| rs.rs_encode_fft(&col.to_vec()))
                .collect::<Result<_, _>>()?;

            #[cfg(not(feature = "parallel"))]
            let encoded_cols = cols
                .iter()
                .map(|col| self.rs.rs_encode_fft(&col.to_vec()))
                .collect::<Result<_, _>>()?;

            encoded_cols
        };

        let encoded_cols_matrix =
            create_matrix!(encoded_cols, encoded_cols.len(), encoded_cols[0].len())?;

        let z_r_2 = encoded_cols_matrix.dot(&self.g_r_2); // X^T.G^T.g' = (GX)^T.g'

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

        #[cfg(feature = "parallel")]
        let commitment_vec = {
            rows_vec
                .par_iter()
                .map(|item| {
                    let mut commitment_scheme = commitment_scheme.clone();
                    let mut serialised_list = Vec::<Vec<u8>>::new();
                    for e in item {
                        let mut writer = Vec::new();
                        e.serialize_uncompressed(&mut writer).unwrap();
                        serialised_list.push(writer);
                    }

                    commitment_scheme.commit(&serialised_list).unwrap().into()
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
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

        #[cfg(feature = "parallel")]
        let commitment_vec = {
            cols_vec
                .par_iter()
                .map(|item| {
                    let mut commitment_scheme = commitment_scheme.clone();
                    let mut serialised_list = Vec::<Vec<u8>>::new();
                    for e in item {
                        let mut writer = Vec::new();
                        e.serialize_uncompressed(&mut writer).unwrap();
                        serialised_list.push(writer);
                    }

                    commitment_scheme.commit(&serialised_list).unwrap().into()
                })
                .collect::<Vec<Vec<u8>>>()
        };

        #[cfg(not(feature = "parallel"))]
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

    fn random_vec(rows: usize, cols: usize) -> Result<Vec<F>, Error> {
        let mut random_vec = Vec::<F>::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            // Create a fresh domain separator for each element
            let domain_separator =
                DomainSeparator::<DefaultHash>::new(TENSOR_VARIANT_DOMAIN_SEPARATOR)
                    .absorb(32, "↪️")
                    .squeeze(16, "↩️");

            let mut prover_state = domain_separator.to_prover_state();

            let random_element = F::rand(prover_state.rng());
            random_vec.push(random_element);

            let mut dest = Vec::new();
            random_element.serialize_uncompressed(&mut dest).unwrap(); // TODO: handle error

            // Absorb the bytes
            prover_state
                .add_bytes(&dest)
                .map_err(|e| Error::Custom(e.to_string()))?;

            // Generate challenge
            let mut chal = [0u8; 16];
            prover_state
                .fill_challenge_bytes(&mut chal)
                .map_err(|e| Error::Custom(e.to_string()))?;
        }

        Ok(random_vec)
    }

    pub fn sample_fft(
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

        if !(w.dot(&self.g_r) == g_s_z_r.slice(s![row_split_start..row_split_end, ..])) {
            // since we rely on fft, we do encoding against the whole matrix and take slice of the result
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

        if !(y.dot(&self.g_r_2)
            == g_prime_s_prime_z_r_prime.slice(s![col_split_start..col_split_end, ..]))
        {
            return Err(Error::Custom(
                "Check 7: (Y^T)S'g¯'r' = G'S'z'r' check failed".to_string(),
            ));
        }

        if !(self.g_r_2.t().dot(&g_s_z_r) == self.g_r.t().dot(&g_prime_s_prime_z_r_prime)) {
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
    use ark_ff::BigInt;
    use ark_serialize::CanonicalSerialize;
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
    fn test_tensor_variant_fft_square_encoding_result() {
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
            variants::tensor_variant_fft::TensorVariantFft::new(ac_commit, n, m).unwrap();

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

            let result = original_grid.variant.sample_fft(
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
    fn test_tensor_variant_fft_row_rec_encoding_result() {
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
            variants::tensor_variant_fft::TensorVariantFft::new(ac_commit, n, m).unwrap();

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

            let result = original_grid.variant.sample_fft(
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
    fn test_tensor_variant_fft_column_rec_encoding_result() {
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
            variants::tensor_variant_fft::TensorVariantFft::new(ac_commit, n, m).unwrap();

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

            let result = original_grid.variant.sample_fft(
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
    fn test_tensor_variant_fft_sampling_failure_on_incorrect_data() {
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
        let tensor_obj =
            variants::tensor_variant_fft::TensorVariantFft::new(ac_commit, n, m).unwrap();

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

            let result = original_grid.variant.sample_fft(
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

    #[test]
    fn test_tensor_variant_fft_commitment_verification() {
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
            variants::tensor_variant_fft::TensorVariantFft::new(ac_commit, 4, 4).unwrap();

        let mut original_grid = DataGrid::new(matrix, tensor_obj).unwrap();
        original_grid.encode().unwrap();

        let cache = original_grid.encode().unwrap();

        // Check that commitments have the expected structure
        assert_eq!(cache.row_wise_commits.len(), cache.z.nrows());
        assert_eq!(cache.col_wise_commits.len(), cache.z.ncols());

        // Check that commitments are not empty
        for commit in &cache.row_wise_commits {
            assert!(!commit.is_empty());
        }
        for commit in &cache.col_wise_commits {
            assert!(!commit.is_empty());
        }
    }
}

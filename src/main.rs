mod commitments;
pub(crate) mod error;
mod grid;
mod variants;

use ark_bls12_381::Fr as Fq;

use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_ff::{BigInt, FftField, Field, inv};
use ark_serialize::CanonicalSerialize;
use ndarray::{arr2, s};
use rand::{Rng, thread_rng};

use crate::{
    commitments::merkle_commitment::{ACMerkleTree, LeafHash, TwoToOneHash},
    grid::DataGrid,
};
fn main() {
    let matrix = arr2(&[
        [Fq::new(BigInt::from(11_u8)), Fq::new(BigInt::from(32_u8))],
        [Fq::new(BigInt::from(1_u8)), Fq::new(BigInt::from(4_u8))],
        [Fq::new(BigInt::from(1_u8)), Fq::new(BigInt::from(3_u8))],
        [Fq::new(BigInt::from(1_u8)), Fq::new(BigInt::from(4_u8))],
    ]);

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
    let i = original_grid.encode().unwrap();

    let vals = original_grid.variant.tensor_cache.as_ref().unwrap();

    let n: usize = original_grid.grid.nrows(); // Number of rows
    let m: usize = original_grid.grid.ncols(); // Number of columns
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

        match result {
            Ok(_) => println!(
                "PASS: rows [{}..{}), cols [{}..{})",
                row_split_start, row_split_end, col_split_start, col_split_end
            ),
            Err(e) => println!(
                "FAIL: rows [{}..{}), cols [{}..{}): {:?}",
                row_split_start, row_split_end, col_split_start, col_split_end, e
            ),
        }
    }
}

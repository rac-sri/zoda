mod commitments;
mod encoding;
pub(crate) mod error;
mod sampling;

use ark_bls12_381::Fr as Fq;

use ark_crypto_primitives::crh::{CRHScheme, TwoToOneCRHScheme};
use ark_ff::{BigInt, FftField, Field};
use ark_serialize::CanonicalSerialize;
use ndarray::arr2;
use rand::{Rng, thread_rng};

use crate::commitments::merkle_commitment::{ACMerkleTree, LeafHash, TwoToOneHash};
fn main() {
    // let matrix = arr2(&[
    //     [
    //         Fq::new(BigInt::from(1_u8)),
    //         Fq::new(BigInt::from(3_u8)),
    //         Fq::new(BigInt::from(4_u8)),
    //         Fq::new(BigInt::from(6_u8)),
    //     ],
    //     [
    //         Fq::new(BigInt::from(1_u8)),
    //         Fq::new(BigInt::from(4_u8)),
    //         Fq::new(BigInt::from(7_u8)),
    //         Fq::new(BigInt::from(8_u8)),
    //     ],
    //     [
    //         Fq::new(BigInt::from(1_u8)),
    //         Fq::new(BigInt::from(3_u8)),
    //         Fq::new(BigInt::from(4_u8)),
    //         Fq::new(BigInt::from(6_u8)),
    //     ],
    //     [
    //         Fq::new(BigInt::from(1_u8)),
    //         Fq::new(BigInt::from(4_u8)),
    //         Fq::new(BigInt::from(7_u8)),
    //         Fq::new(BigInt::from(8_u8)),
    //     ],
    // ]);
    // let original_grid = encoding::DataGrid::new(4_u8, 4_u8, matrix).unwrap();

    // let mut leaves: Vec<Vec<u8>> = Vec::new();
    // for _ in 0..4 {
    //     let fq = Fq::new(BigInt::from(1_u8));
    //     let mut buf = Vec::new();
    //     fq.serialize_uncompressed(&mut buf).unwrap();
    //     leaves.push(buf);
    // }

    let mut rng = thread_rng();
    let tensor_obj = encoding::tensor_variant::TensorVariant::new();

    // let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
    // let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

    // let ac_merkle_tree = ACMerkleTree::new(leaf_crh_params, two_to_one_crh_params, leaves).unwrap();

    // let encoding = tensor_obj
    //     .encode(original_grid.grid, ac_merkle_tree)
    //     .unwrap();

    // println!("matrix: {:?} ", encoding.0);

    // println!("\n \n \n \n \n \n \n Next \n \n \n \n ");
    let matrix = arr2(&[
        [
            Fq::new(BigInt::from(1_u8)),
            Fq::new(BigInt::from(3_u8)),
            Fq::new(BigInt::from(4_u8)),
            Fq::new(BigInt::from(6_u8)),
        ],
        [
            Fq::new(BigInt::from(1_u8)),
            Fq::new(BigInt::from(4_u8)),
            Fq::new(BigInt::from(7_u8)),
            Fq::new(BigInt::from(8_u8)),
        ],
        [
            Fq::new(BigInt::from(1_u8)),
            Fq::new(BigInt::from(3_u8)),
            Fq::new(BigInt::from(8_u8)),
            Fq::new(BigInt::from(5_u8)),
        ],
        [
            Fq::new(BigInt::from(1_u8)),
            Fq::new(BigInt::from(2_u8)),
            Fq::new(BigInt::from(6_u8)),
            Fq::new(BigInt::from(8_u8)),
        ],
    ]);

    let mut leaves: Vec<Vec<u8>> = Vec::new();
    for _ in 0..4 {
        let fq = Fq::new(BigInt::from(1_u8));
        let mut buf = Vec::new();
        fq.serialize_uncompressed(&mut buf).unwrap();
        leaves.push(buf);
    }

    let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
    let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

    let original_grid = encoding::DataGrid::new(4_u8, 4_u8, matrix).unwrap();
    let ac_merkle_tree = ACMerkleTree::new(leaf_crh_params, two_to_one_crh_params, leaves).unwrap();

    println!("{:?}", Fq::GENERATOR);
    let vandermonte_matrix = tensor_obj
        .encode(original_grid.grid, ac_merkle_tree, Fq::GENERATOR)
        .unwrap();
}

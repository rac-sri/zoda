use crate::commitments::ACCommitmentScheme;
use ark_crypto_primitives::{
    crh::{
        injective_map::{PedersenCRHCompressor, PedersenTwoToOneCRHCompressor, TECompressor},
        {CRHScheme, TwoToOneCRHScheme, pedersen},
    },
    merkle_tree::{ByteDigestConverter, Config, MerkleTree, Path},
};
use ark_ed_on_bls12_381::EdwardsProjective as CurveGroup;
use ark_serialize::CanonicalSerialize;

pub type TwoToOneHash = PedersenTwoToOneCRHCompressor<CurveGroup, TECompressor, TwoToOneWindow>;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TwoToOneWindow;

// `WINDOW_SIZE * NUM_WINDOWS` = 2 * 256 bits = enough for hashing two outputs.
impl pedersen::Window for TwoToOneWindow {
    const WINDOW_SIZE: usize = 4;
    const NUM_WINDOWS: usize = 128;
}

pub type LeafHash = PedersenCRHCompressor<CurveGroup, TECompressor, LeafWindow>;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LeafWindow;

// `WINDOW_SIZE * NUM_WINDOWS` = 2 * 256 bits = enough for hashing two outputs.
impl pedersen::Window for LeafWindow {
    const WINDOW_SIZE: usize = 4;
    const NUM_WINDOWS: usize = 144;
}

#[derive(Clone)]
pub struct MerkleConfig;
impl Config for MerkleConfig {
    // Our Merkle tree relies on two hashes: one to hash leaves, and one to hash pairs
    // of internal nodes.
    type Leaf = [u8];
    type LeafHash = LeafHash;
    type TwoToOneHash = TwoToOneHash;
    type LeafDigest = <LeafHash as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <TwoToOneHash as TwoToOneCRHScheme>::Output;
}

pub type ACMerkleTree = MerkleTree<MerkleConfig>;

impl ACCommitmentScheme<Vec<u8>, Vec<u8>> for ACMerkleTree {
    fn commit(&self, _items: &Vec<Vec<u8>>) -> Vec<u8> {
        let mut writer = Vec::new();
        self.root().serialize_uncompressed(&mut writer).unwrap();
        writer
    }

    fn proof(&self, items: &Vec<Vec<u8>>) -> Vec<u8> {
        let mut writer = Vec::new();
        self.generate_proof(0);
        writer
    }

    fn verify(&self, items: &Vec<Vec<u8>>, commitments: &Vec<Vec<u8>>) -> bool {
        true
    }

    fn open(
        &self,
        items: &Vec<Vec<u8>>,
        commitments: &Vec<Vec<u8>>,
        proof: &Vec<Vec<u8>>,
    ) -> Vec<u8> {
        vec![]
    }
}

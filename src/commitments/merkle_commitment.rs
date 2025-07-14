use crate::{commitments::ACCommitmentScheme, error::CommitmentError};
use ark_crypto_primitives::{
    crh::{
        injective_map::{PedersenCRHCompressor, PedersenTwoToOneCRHCompressor, TECompressor},
        {CRHScheme, TwoToOneCRHScheme, pedersen},
    },
    merkle_tree::{ByteDigestConverter, Config, MerkleTree, Path},
};
use ark_ed_on_bls12_381::EdwardsProjective as CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::thread_rng;

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
    type Leaf = [u8];
    type LeafHash = LeafHash;
    type TwoToOneHash = TwoToOneHash;
    type LeafDigest = <LeafHash as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <TwoToOneHash as TwoToOneCRHScheme>::Output;
}

pub type ACMerkleTree = MerkleTree<MerkleConfig>;

impl ACCommitmentScheme<Vec<u8>> for ACMerkleTree {
    type Commitment = Vec<u8>;
    type Proof = Result<Path<MerkleConfig>, CommitmentError>;
    type Opening = Result<bool, CommitmentError>;

    fn commit(&mut self, _items: &Vec<u8>) -> Self::Commitment {
        let mut writer = Vec::new();
        self.root().serialize_uncompressed(&mut writer).unwrap();
        writer
    }

    fn proof(&self, index: &Vec<u8>) -> Self::Proof {
        self.generate_proof(usize::from_le_bytes(index.as_slice().try_into().unwrap()))
            .map_err(|_| CommitmentError::ProofGenerationError)
    }

    fn open(&self, leaf: &Vec<u8>, path: &Vec<u8>) -> Self::Opening {
        let path = Path::<MerkleConfig>::deserialize_uncompressed(path.as_slice())
            .map_err(|_| CommitmentError::PathDeserialisationFailed)?;

        let mut rng = thread_rng();
        let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
        let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

        path.verify(
            &leaf_crh_params,
            &two_to_one_crh_params,
            &self.root(),
            leaf.as_slice(),
        )
        .map_err(|_| CommitmentError::ProofGenerationError)
    }
}

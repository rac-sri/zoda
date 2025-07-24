use crate::{commitments::ACCommitmentScheme, error::CommitmentError};
use ark_crypto_primitives::{
    crh::{
        CRHScheme, TwoToOneCRHScheme,
        injective_map::{PedersenCRHCompressor, PedersenTwoToOneCRHCompressor, TECompressor},
        pedersen,
    },
    merkle_tree::{ByteDigestConverter, Config, MerkleTree, Path},
};
use ark_ed_on_bls12_381::EdwardsProjective as CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::thread_rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

#[derive(Clone)]
pub struct ACMerkleTree {
    tree: MerkleTree<MerkleConfig>,
    pub leaf_hash_param: <LeafHash as CRHScheme>::Parameters,
    pub two_to_one_hash_param: <TwoToOneHash as TwoToOneCRHScheme>::Parameters,
}

impl ACMerkleTree {
    pub fn new<
        L: AsRef<<MerkleConfig as ark_crypto_primitives::merkle_tree::Config>::Leaf> + Send,
    >(
        leaf_hash_param: <LeafHash as CRHScheme>::Parameters,
        two_to_one_hash_param: <TwoToOneHash as TwoToOneCRHScheme>::Parameters,
        #[cfg(not(feature = "parallel"))] leaves: impl IntoIterator<Item = L>,
        #[cfg(feature = "parallel")] leaves: impl IntoParallelIterator<Item = L>,
    ) -> Result<Self, CommitmentError> {
        let tree =
            MerkleTree::<MerkleConfig>::new(&leaf_hash_param, &two_to_one_hash_param, leaves)
                .map_err(|e| CommitmentError::Custom(e.to_string()))?;

        Ok(Self {
            tree,
            leaf_hash_param,
            two_to_one_hash_param,
        })
    }

    fn construct_tree<
        L: AsRef<<MerkleConfig as ark_crypto_primitives::merkle_tree::Config>::Leaf> + Send,
    >(
        &mut self,
        #[cfg(not(feature = "parallel"))] leaves: impl IntoIterator<Item = L>,
        #[cfg(feature = "parallel")] leaves: impl IntoParallelIterator<Item = L>,
    ) -> Result<MerkleTree<MerkleConfig>, CommitmentError> {
        let tree = MerkleTree::<MerkleConfig>::new(
            &self.leaf_hash_param,
            &self.two_to_one_hash_param,
            leaves,
        )
        .map_err(|e| CommitmentError::Custom(e.to_string()))?;

        Ok(tree)
    }
}

impl ACCommitmentScheme<Vec<Vec<u8>>, Vec<Vec<u8>>> for ACMerkleTree {
    type Commitment = Vec<u8>;
    type Proof = Path<MerkleConfig>;
    type Opening = Result<bool, CommitmentError>;

    fn commit(&mut self, items: &Vec<Vec<u8>>) -> Result<Self::Commitment, CommitmentError> {
        #[cfg(feature = "parallel")]
        let tree = self.construct_tree(items.par_iter())?;

        #[cfg(not(feature = "parallel"))]
        let tree = self.construct_tree(items.iter())?;

        let mut writer = Vec::new();
        tree.root().serialize_uncompressed(&mut writer).unwrap();
        Ok(writer)
    }

    fn proof(&self, index: &Vec<Vec<u8>>) -> Result<Self::Proof, CommitmentError> {
        if index.len() != 1 {
            CommitmentError::Custom("Invalid Leaf Input".to_string());
        }
        self.tree
            .generate_proof(usize::from_le_bytes(
                index[0].as_slice().try_into().unwrap(),
            ))
            .map_err(|_| CommitmentError::ProofGenerationError)
    }

    fn open(
        &self,
        leaf: &Vec<Vec<u8>>,
        path: &Vec<Vec<u8>>,
    ) -> Result<Self::Opening, CommitmentError> {
        if leaf.len() != 1 {
            CommitmentError::Custom("Invalid Leaf Input".to_string());
        }
        let flat_path: Vec<u8> = path.iter().flat_map(|v| v.iter()).copied().collect();
        let path = Path::<MerkleConfig>::deserialize_uncompressed(flat_path.as_slice())
            .map_err(|_| CommitmentError::PathDeserialisationFailed)?;

        let mut rng = thread_rng();
        let leaf_crh_params = <LeafHash as CRHScheme>::setup(&mut rng).unwrap();
        let two_to_one_crh_params = <TwoToOneHash as TwoToOneCRHScheme>::setup(&mut rng).unwrap();

        Ok(path
            .verify(
                &leaf_crh_params,
                &two_to_one_crh_params,
                &self.tree.root(),
                leaf[0].as_slice(),
            )
            .map_err(|_| CommitmentError::ProofGenerationError))
    }

    fn verify(&self, point: &Vec<Vec<u8>>, proof: &Vec<Vec<u8>>) -> Result<bool, CommitmentError> {
        Ok(self.open(point, proof).is_ok())
    }
}

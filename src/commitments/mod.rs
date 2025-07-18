use crate::error::CommitmentError;

pub mod merkle_commitment;
pub trait ACCommitmentScheme<P, T> {
    type Commitment;
    type Proof;
    type Opening;

    fn commit(&mut self, items: &P) -> Result<Self::Commitment, CommitmentError>;
    fn proof(&self, items: &P) -> Result<Self::Proof, CommitmentError>;
    fn open(&self, items: &P, proof: &T) -> Result<Self::Opening, CommitmentError>;
    fn verify(&self, items: &P, proof: &T) -> Result<bool, CommitmentError>;
}

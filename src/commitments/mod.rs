use crate::error::CommitmentError;

pub mod merkle_commitment;
pub trait ACCommitmentScheme<T: AsRef<[u8]>> {
    type Commitment;
    type Proof;
    type Opening;

    fn commit(&mut self, items: &T) -> Self::Commitment;
    fn proof(&self, items: &T) -> Self::Proof;
    fn open(&self, items: &T, proof: &T) -> Self::Opening;
    fn verify(&self, items: &T, proof: &T) -> bool;
}

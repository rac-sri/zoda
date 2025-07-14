pub mod merkle_commitment;

pub trait ACCommitmentScheme<T: AsRef<[u8]>, S> {
    fn commit(&self, items: &Vec<T>) -> S;
    fn proof(&self, items: &Vec<T>) -> S;
    fn verify(&self, items: &Vec<T>, commitments: &Vec<T>) -> bool;
    fn open(&self, items: &Vec<T>, commitments: &Vec<T>, proof: &Vec<T>) -> S;
}

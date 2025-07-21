use crate::{commitments::ACCommitmentScheme, error::Error};

pub mod reed_solomon;
pub mod tensor_variant;

pub(crate) trait Variant<P> {
    fn encode(&mut self, values: &P) -> Result<P, Error>;
    fn decode(&mut self, original: &P, shards: &P) -> Result<P, Error>;
}

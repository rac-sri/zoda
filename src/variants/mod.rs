use crate::error::Error;

pub mod reed_solomon;
pub mod tensor_variant;
pub mod tensor_variant_fft;

pub(crate) trait Variant<P> {
    fn encode(&mut self, values: &P) -> Result<P, Error>;
    fn decode(&mut self, original: &P, shards: &P) -> Result<P, Error>;
}

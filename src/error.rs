#[derive(Debug)]
pub enum Error {
    LengthMismatch,
    Custom(String),
    MatrixShapeError(String),
    EncodingError,
    DivisorIsZero,
    MatrixDimsMismatch,
}

#[derive(Debug)]
pub enum CommitmentError {
    ProofGenerationError,
    ProofVerificationError,
    PathDeserialisationFailed,
    Custom(String),
}

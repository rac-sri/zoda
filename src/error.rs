pub enum Error {
    LengthMismatch,
    Custom(String),
    MatrixShapeError(String),
    EncodingError,
}

pub enum CommitmentError {
    ProofGenerationError,
    ProofVerificationError,
    PathDeserialisationFailed,
}

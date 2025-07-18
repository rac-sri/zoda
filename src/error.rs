pub enum Error {
    LengthMismatch,
    Custom(String),
    MatrixShapeError(String),
    EncodingError,
    DivisorIsZero,
    MatrixDimsMismatch,
}

pub enum CommitmentError {
    ProofGenerationError,
    ProofVerificationError,
    PathDeserialisationFailed,
}

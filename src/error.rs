pub enum Error {
    LengthMismatch,
    Custom(String),
    MatrixShapeError(String),
    EncodingError,
}

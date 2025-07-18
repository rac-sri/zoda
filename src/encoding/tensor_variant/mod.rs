use crate::{commitments::ACCommitmentScheme, encoding::reed_solomon::ReedSolomon, error::Error};
use ark_ff::Field;
use ndarray::{Array2 as Matrix, ArrayBase};
use rand::Rng;
use spongefish::{BytesToUnitSerialize, DefaultHash, DomainSeparator, UnitToBytes, *};
const TENSOR_VARIANT_DOMAIN_SEPARATOR: &str = "ZODA-TENSOR-VARIANT";
struct TensorVariant<C: ACCommitmentScheme<Vec<u8>>> {
    rs: ReedSolomon,
    commitment: C,
}

impl<C: ACCommitmentScheme<Vec<u8>>> TensorVariant<C> {
    pub fn new(commitment: C) -> Self {
        let rs = ReedSolomon::new(2);
        Self { rs, commitment }
    }

    pub fn encode<F: Field>(
        &self,
        original_grid: Matrix<F>,
    ) -> Result<
        (
            Matrix<F>,
            Matrix<F>,
            Matrix<F>,
            Matrix<F>,
            (Vec<u8>, Matrix<F>),
            (Vec<u8>, Matrix<F>),
        ),
        Error,
    > {
        // calculate Z = GXG'
        let Z = self.rs.tensor_encode(&original_grid)?;
        let (G, G_2) = self
            .rs
            .get_generator_matrixes::<F>(original_grid.nrows(), original_grid.ncols())?;

        if original_grid.nrows() != original_grid.ncols() {
            return Err(Error::MatrixDimsMismatch);
        }

        // generate \tilde_g and \tilde_g'
        let tilde_g_r =
            self.diagnol_matrix_gen::<F>(original_grid.nrows(), original_grid.ncols())?;
        let tilde_g_r_2 =
            self.diagnol_matrix_gen::<F>(original_grid.ncols(), original_grid.ncols())?;

        let z_r = original_grid.dot(&G_2).dot(&tilde_g_r.1);
        let z_r_2 = original_grid.t().dot(&G.t()).dot(&tilde_g_r_2.1);
        Ok((original_grid, Z, z_r, z_r_2, tilde_g_r, tilde_g_r_2))
    }

    fn diagnol_matrix_gen<F: Field>(
        &self,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, Matrix<F>), Error> {
        let domain_separator = DomainSeparator::<DefaultHash>::new("example-protocol 🤌")
            .absorb(1, "↪️")
            .squeeze(16, "↩️");
        let mut prover_state = domain_separator.to_prover_state();

        let mut diagnoal_matrix = Matrix::<F>::zeros((rows, cols));
        // Since we are enforcing a square matrix
        if rows != cols {
            return Err(Error::MatrixShapeError(format!(
                "Rows {} : Cols {}",
                rows, cols
            )));
        }
        for i in 0..rows {
            let mut private = [0u8; 32];
            prover_state.rng().fill(&mut private);

            diagnoal_matrix[(i, i)] = F::from_random_bytes(private.as_ref()).ok_or_else(|| {
                Error::Custom("Failed to convert random bytes to field element".to_string())
            })?;
            prover_state
                .add_bytes(private.as_ref())
                .map_err(|e| Error::Custom(e.to_string()));
            // The prover receive a 128-bit challenge.
            let mut chal = [0u8; 16];
            prover_state.fill_challenge_bytes(&mut chal).unwrap();
        }

        Ok((prover_state.narg_string().to_vec(), diagnoal_matrix))
    }
}

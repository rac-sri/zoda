use crate::error::Error;
use ark_ff::{FftField, fields::Field};
use ark_poly::{
    DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, univariate::DensePolynomial,
};
use ndarray::{Array2 as Matrix, s};

#[macro_export]
macro_rules! create_matrix {
    ($entries:expr, $rows:expr, $cols:expr) => {
        Matrix::from_shape_vec(
            ($rows.into(), $cols.into()),
            $entries.into_iter().flatten().collect(),
        )
        .map_err(|e| Error::MatrixShapeError(e.to_string()))
    };
}

pub struct ReedSolomon {}

impl ReedSolomon {
    pub fn vandermonde_matrix<F: Field>(
        &self,
        alphas: &Vec<F>,
        k: usize,
        n: usize,
    ) -> Result<Matrix<F>, Error> {
        if alphas.len() != n {
            return Err(Error::Custom(format!(
                "alphas length {} must match codeword length n {}",
                alphas.len(),
                n
            )));
        }
        if k > n {
            return Err(Error::Custom(
                "Message length k cannot be greater than codeword length n".to_string(),
            ));
        }

        // Build the full k x n Vandermonde matrix V.
        let vandermonde_rows: Vec<Vec<F>> = (0..k)
            .map(|i| alphas.iter().map(|alpha| alpha.pow([i as u64])).collect())
            .collect();
        let vandermonde_matrix = create_matrix!(vandermonde_rows, k, n)?;

        // Take the first k alphas to form the square Vandermonde matrix V_sys,
        // which is the first k columns of V.
        let sys_alphas: Vec<F> = alphas.iter().take(k).cloned().collect();

        // Invert V_sys using your existing Lagrange-based inversion method.
        let v_sys_inv = self.invert_vandermonde_matrix(&sys_alphas)?;

        // The systematic generator matrix G = V_sys_inv * V.
        Ok(v_sys_inv.dot(&vandermonde_matrix).t().to_owned())
    }

    pub fn invert_vandermonde_matrix<F: Field>(&self, alphas: &Vec<F>) -> Result<Matrix<F>, Error> {
        let n = alphas.len();
        let mut inv = vec![vec![F::zero(); n]; n];

        for j in 0..n {
            // Compute the Lagrange polynomial P_j(x) = product_{m≠j} (x - alpha_m) / (alpha_j - alpha_m)
            // First, compute the denominator: product_{m≠j} (alpha_j - alpha_m)
            let mut denom = F::one();
            for m in 0..n {
                if m != j {
                    denom *= alphas[j] - alphas[m];
                }
            }
            let denom_inv = denom.inverse().unwrap();

            // Build the numerator polynomial: product_{m≠j} (x - alpha_m)
            let mut numerator: DensePolynomial<F> =
                DensePolynomial::<F>::from_coefficients_vec(vec![F::one()]); // Start with 1

            for m in 0..n {
                if m != j {
                    // Multiply by (x - alpha_m)
                    let factor = DensePolynomial::from_coefficients_vec(vec![-alphas[m], F::one()]);
                    numerator = numerator.naive_mul(&factor);
                }
            }

            // Divide by denominator and store coefficients in the j-th row
            for k in 0..n {
                if k < numerator.coeffs.len() {
                    inv[j][k] = numerator.coeffs[k] * denom_inv;
                } else {
                    inv[j][k] = F::zero();
                }
            }
        }

        create_matrix!(inv, inv.len(), inv[0].len())
    }
    pub fn alphas_with_generator<F: Field>(&self, n: usize, generator: F) -> Vec<F> {
        (1..=n).map(|i| generator.pow([i as u64])).collect()
    }

    pub fn rs_encode<F: Field>(&self, msg: &Matrix<F>, G: &Matrix<F>) -> Result<Matrix<F>, Error> {
        Ok(msg.dot(G))
    }

    pub fn rs_encode_fft<F: FftField>(&self, points: &Vec<F>) -> Result<Vec<F>, Error> {
        let domain = GeneralEvaluationDomain::<F>::new(points.len())
            .ok_or_else(|| Error::Custom("DOMAIN GENERATION ERROR".to_string()))?;

        Ok(domain.fft(points))
    }
}

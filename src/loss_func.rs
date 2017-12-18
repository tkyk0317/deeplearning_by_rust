extern crate nalgebra;

use nalgebra::core::{DMatrix};

// 損失関数.
pub fn sum_of_square(result: &DMatrix<f64>, reference: &DMatrix<f64>) -> f64 {
    result.iter()
          .zip(reference.iter())
          .fold(0., |acc, (_result, _reference)| {
              acc + (_result - _reference).powi(2)
          }) * 0.5
}

// 交差エントロピー関数.
pub fn cross_entropy(result: &DMatrix<f64>, reference: &DMatrix<f64>) -> f64 {
    -1. * result.iter()
                .zip(reference.iter())
                .fold(0., |acc, (_result, _reference)| {
                    // マイナス無限大が発生しないよう、小さな値を加算(ln(0)＝-inf).
                    acc + _reference * (_result + 1.0e-7).ln()
                })
}

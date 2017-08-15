extern crate nalgebra;

use nalgebra::core::{DMatrix};

// ニューロン構造体.
pub struct Neuron<'a> {
    bias: &'a DMatrix<f64>,
    data: &'a DMatrix<f64>,
    weight: &'a DMatrix<f64>,
}

// ニューロン実装.
impl<'a> Neuron<'a> {
    pub fn new(bias: &'a DMatrix<f64>, data: &'a DMatrix<f64>, weight: &'a DMatrix<f64>) -> Self {
        Neuron { bias: bias, data: data, weight: weight }
    }

    // 内積.
    fn dot(&self) -> DMatrix<f64> {
        self.data * self.weight + self.bias
    }

    // ステップ関数.
    pub fn step(&self) -> DMatrix<f64> {
        self.dot().map(|i| {
            match i > 0. {
                true => 1.,
                false => 0.
            }
        })
    }

    // シグモイド関数.
    pub fn sigmoid(&self) -> DMatrix<f64> {
        self.dot().map(|i| { 1. / (1. + (-i).exp()) })
    }

    // ReLU関数.
    pub fn relu(&self) -> DMatrix<f64> {
        self.dot().map(|i| i.max(0.))
    }
}
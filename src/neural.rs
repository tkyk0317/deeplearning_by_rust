extern crate nalgebra;

use neuron;
use gradient;
use loss_func;
use nalgebra::core::{DMatrix};

// ニューラルネットワーク.
pub struct NeuralNW<'a> {
    data: &'a DMatrix<f64>,
    reference: &'a DMatrix<f64>,
}

impl<'a> NeuralNW<'a> {
    pub fn new(data: &'a DMatrix<f64>, reference: &'a DMatrix<f64>) -> Self {
        NeuralNW{ data: data, reference: reference }
    }

    /// トレーニング.
    /// 重み、バイアスを変化させながら、LossFuncを実行.
    pub fn trainning(&self) {
        let mut x = DMatrix::<f64>::from_iterator(1, 3, [0.1, 0.2, 0.3].iter().cloned());
        gradient::GradientDescent::gradient(self, &mut x);
    }

    /// 勾配法コールバック関数.
    /// この関数で、ニューラルネットワークを実施する.
    ///
    /// ## 引数.
    /// &DMatrix<f64> 勾配法により変化させるパラメータ.
    //fn callback(d: &DMatrix<f64>, x: &DMatrix<f64>) -> f64 {
    pub fn callback(&self, d: &DMatrix<f64>) -> f64 {
        let b1 = DMatrix::<f64>::from_element(1, d.ncols(), 0.);
        let n1 = neuron::Neuron::new(&b1, &d, &self.data);
        let x2 = n1.sigmoid();
        return loss_func::cross_entropy(&x2, self.reference);
    }
}
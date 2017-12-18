extern crate nalgebra;

use neuron;
use gradient;
use loss_func;
use nalgebra::core::{DMatrix};
use std::rc::Rc;
use std::cell::RefCell;
use std::vec::Vec;

// エイリアス指定.
type InputData = Rc<RefCell<DMatrix<f64>>>;
type BiasData = Vec<Rc<RefCell<DMatrix<f64>>>>;
type WeightData = Vec<Rc<RefCell<DMatrix<f64>>>>;
type ReferenceData = Rc<RefCell<DMatrix<f64>>>;

/// 勾配法コールバック関数.
/// この関数で、ニューラルネットワークを実施する.
///
/// ## 引数.
/// &DMatrix<f64> 勾配法により変化させるパラメータ.
//fn callback(d: &DMatrix<f64>, x: &DMatrix<f64>) -> f64 {
fn callback_bias(obj :&NeuralNW, d: &DMatrix<f64>) -> f64 {
    let weight = &*obj.input_weight[0].borrow();
    let data = &*obj.input_data.borrow();
    let n1 = neuron::Neuron::new(d, data, weight);
    let x2 = n1.sigmoid();
    return loss_func::cross_entropy(&x2, &*obj.reference.borrow());
}
fn callback_weight(obj :&NeuralNW, d: &DMatrix<f64>) -> f64 {
    let bias = &*obj.input_bias[0].borrow();
    let data = &*obj.input_data.borrow();
    let n1 = neuron::Neuron::new(bias, data, d);
    let x2 = n1.sigmoid();
    return loss_func::cross_entropy(&x2, &*obj.reference.borrow());
}

// ニューラルネットワーク.
pub struct NeuralNW {
    //pub input_data: Rc<RefCell<DMatrix<f64>>>,        // 入力データ.
    pub input_data: InputData,
    pub input_bias: BiasData,
    pub input_weight: WeightData,
    pub reference: ReferenceData,
}

impl NeuralNW {
    pub fn new(input_data: InputData,
               input_bias: BiasData,
               input_weight: WeightData,
               reference: ReferenceData) -> Self {
        NeuralNW{ input_data: input_data, input_bias: input_bias,
                  input_weight: input_weight, reference: reference }
    }

    /// トレーニング.
    /// 重み、バイアスを変化させながら、LossFuncを実行.
    pub fn trainning(&self) {
        let _bias = gradient::GradientDescent::gradient(self, &self.input_bias[0], callback_bias);
        let _weight = gradient::GradientDescent::gradient(self, &self.input_weight[0], callback_weight);
    }
}


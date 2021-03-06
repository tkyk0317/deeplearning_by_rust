extern crate gnuplot;
extern crate nalgebra;
extern crate mnist;

use nalgebra::core::{DMatrix};
use mnist::{Mnist, MnistBuilder};
use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use std::vec::Vec;
use std::rc::Rc;
use std::cell::RefCell;

mod neuron;
mod neural;
mod gradient;
mod loss_func;

// 活性化関数描画.
#[test]
fn draw_active_function() {
    // ステップ関数.
    fn step(x: f64) -> u64 {
        match x > 0. {
            true  => 1,
            false => 0,
        }
    }
    // シグモイド関数.
    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }
    // relu関数.
    fn relu(x: f64) -> f64 {
        x.max(0.)
    }

    // 活性化関数適用.
    let mut x: Vec<f64> = std::vec::Vec::new();
    for i in -100..100 { x.push((i as f64) / 10.); }
    let sig: Vec<f64> = x.iter().map(|i| sigmoid(*i)).collect(); // sigmoid function.
    let step: Vec<u64> = x.iter().map(|i| step(*i)).collect(); // step function.
    let relu: Vec<f64> = x.iter().map(|i| relu(*i)).collect(); // relu function.

    // グラフ描画.
    let mut fg = Figure::new();
    fg.axes2d()
      .lines(&x, &sig,  &[Caption("sigmoid"), Color("red")])
      .lines(&x, &step, &[Caption("step"), Color("blue")])
      .lines(&x, &relu, &[Caption("relu"), Color("green")])
      .set_y_range(Fix(-0.5), Fix(2.0));
    fg.show();
}

// 1層ニューラルネットワーク_
fn single_neural_nw() {
    // 第一層入力データ.
    let b1 = DMatrix::<f64>::from_iterator(1, 3, [0., 0., 0.].iter().cloned());
    let x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 2.].iter().cloned());
    let w1 = DMatrix::<f64>::from_iterator(2, 3, [1., 2., 3., 4., 5., 6.].iter().cloned());
    let n1 = neuron::Neuron::new(&b1, &x1, &w1);

    // 出力層.
    println!("identify:\n{}", n1.identify());
}

// 多層ニューラルネットワーク.
fn multi_neural_nw() {
    let b1 = DMatrix::<f64>::from_iterator(1, 3, [0.1, 0.2, 0.3].iter().cloned());
    let x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 0.5].iter().cloned());
    let w1 = DMatrix::<f64>::from_iterator(2, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().cloned());
    let n1 = neuron::Neuron::new(&b1, &x1, &w1);

    let b2 = DMatrix::<f64>::from_iterator(1, 2, [0.1, 0.2].iter().cloned());
    let w2 = DMatrix::<f64>::from_iterator(3, 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().cloned());
    let x2 = n1.sigmoid();
    let n2 = neuron::Neuron::new(&b2, &x2, &w2);

    let b3 = DMatrix::<f64>::from_iterator(1, 2, [0.1, 0.2].iter().cloned());
    let w3 = DMatrix::<f64>::from_iterator(2, 2, [0.1, 0.2, 0.3, 0.4].iter().cloned());
    let x3 = n2.sigmoid();
    let n3 = neuron::Neuron::new(&b3, &x3, &w3);

    println!("softmax:\n{}", n3.softmax());
}

// mnistニューラルネットワーク.
fn mnist_neural_nw() {
    let (size, rows, cols) = (50_000, 28, 28);
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new().label_format_digit()
                                                            .training_set_length(size)
                                                            //.validation_set_length(size)
                                                            //.test_set_length(size)
                                                            .finalize();
    let _x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 0.5].iter().cloned());
    let x1 = Rc::new(RefCell::new(_x1));
    let _r1 = DMatrix::<f64>::from_iterator(1, trn_img.len(), trn_img.iter().map(|i| *i as f64));
    let r1 = Rc::new(RefCell::new(_r1));
    let _b1 = DMatrix::<f64>::from_iterator(1, 3, [0.00001, 0.00002, 0.00003].iter().cloned());
    let b1 = vec![Rc::new(RefCell::new(_b1))];
    let _w1 = DMatrix::<f64>::from_iterator(2, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].iter().cloned());
    let w1 = vec![Rc::new(RefCell::new(_w1))];

    // ニューラルネットワーク実施.
    let nw = neural::NeuralNW::new(x1, b1, w1, r1);
    nw.trainning();
}

// main関数.
fn main() {
    // 一層ニューラルネットワーク.
    single_neural_nw();

    // 多層ニューラルネットワーク.
    multi_neural_nw();

    // mnistニューラルネットワーク.
    mnist_neural_nw();
}

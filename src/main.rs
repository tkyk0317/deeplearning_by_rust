extern crate gnuplot;
extern crate nalgebra;

use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use nalgebra::core::{DMatrix};

// ニューロン構造体.
struct Neuron<'a> {
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
}

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

// 活性化関数描画.
fn draw_active_function() {
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

// main関数.
fn main() {
    // 第一層入力データ.
    let b1 = DMatrix::<f64>::from_iterator(1, 3, [0., 0., 0.].iter().cloned());
    let x1 = DMatrix::<f64>::from_iterator(1, 2, [1., 2.].iter().cloned());
    let w1 = DMatrix::<f64>::from_iterator(2, 3, [1., 2., 3., 4., 5., 6.].iter().cloned());
    let n1 = Neuron::new(&b1, &x1, &w1);

    // 出力層.
    println!("dot: {}", n1.dot());

    let b2 = DMatrix::<f64>::from_iterator(2, 3, [0., 0., 0., 0., 0., 0.].iter().cloned());
    let x2 = DMatrix::<f64>::from_iterator(2, 6, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.].iter().cloned());
    let w2 = DMatrix::<f64>::from_iterator(6, 3, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.].iter().cloned());
    let n2 = Neuron::new(&b2, &x2, &w2);
    println!("dot: {}", n2.dot());

    // グラフ描画.
    //draw_active_function();
}

extern crate gnuplot;
extern crate ndarray;

use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use ndarray::{Array1, Array2};

// ニューロン構造体.
struct Neuron<'a> {
    bias: f64,
    data: &'a Array2<f64>,
    weight: &'a Array2<f64>,
}

// ニューロン実装.
impl<'a> Neuron<'a> {
    pub fn new(bias: f64, data: &'a Array2<f64>, weight: &'a Array2<f64>) -> Self {
        Neuron { bias: bias, data: data, weight: weight }
    }

    // 内積.
    fn dot(&self) -> Array2<f64> {
        self.data.dot(self.weight) + self.bias
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
    let x1 = Array2::<f64>::from(vec![[1., 2.]]);
    let w1 = Array2::<f64>::from(vec![[1. ,3., 5.], [2., 4., 6.]]);
    let n1 = Neuron::new(0., &x1, &w1);
    // 出力層.
    println!("dot: {}", n1.dot());

    // グラフ描画.
    //draw_active_function();
}

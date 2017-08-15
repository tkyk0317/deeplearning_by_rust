extern crate gnuplot;
extern crate ndarray;

use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use ndarray::{Array1};

// ニューロン構造体.
struct Neuron {
    bias: f64,
    data: Array1<f64>,
    weight: Array1<f64>,
}

// ニューロン実装.
impl Neuron {
    pub fn new(bias: f64, data: Array1<f64>, weight: Array1<f64>) -> Option<Neuron> {
        if data.len() != weight.len() {
            return None;
        }
        return Some(Neuron { bias: bias, data: data, weight: weight });
    }

    // 内積.
    fn dot(&self) -> f64 {
        self.data.dot(&self.weight) + self.bias
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
    match x > 0. {
        true  => x,
        false => 0.
    }
}

fn main() {
    let mut n = Array1::<f64>::zeros(4);
    let mut w = Array1::<f64>::zeros(4);
    n[0] = 10.;
    w[0] = 2.;
    let _n = Neuron::new(10., n, w).unwrap();

    println!("{:?}", _n.dot());
    println!("{}", sigmoid(_n.dot()));

    // 活性化関数適用.
    let mut x: Vec<f64> = std::vec::Vec::new();
    for i in -100..100 { x.push((i as f64) / 10.); }
    let sig: Vec<f64> = x.iter().map(|i| sigmoid(*i)).collect(); // sigmoid function.
    let step: Vec<u64> = x.iter().map(|i| step(*i)).collect(); // step function.
    let relu: Vec<f64> = x.iter().map(|i| relu(*i)).collect(); // relu function.

    // グラフ描画.
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(&x, &sig,  &[Caption("Sigmoid"), Color("red")])
        .lines(&x, &step, &[Caption("Step"), Color("blue")])
        .lines(&x, &relu, &[Caption("ReLU"), Color("green")])
        .set_y_range(Fix(-0.5), Fix(2.0));
    fg.show();
}

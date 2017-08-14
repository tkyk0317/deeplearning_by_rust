extern crate gnuplot;
extern crate ndarray;

use gnuplot::{Figure, Caption, Color};
use ndarray::{Array1};

// ニューロン構造体.
struct Neuron {
    bias: f64,
    data: Array1<f64>,
    weight: Array1<f64>,
}

// ニューロン実装.
impl Neuron {
    pub fn new(bias: f64, data: Array1<f64>, weight: Array1<f64>) -> Neuron {
        Neuron { bias: bias, data: data, weight: weight }
    }

    // パーセプトロン.
    fn perceptron(&self) -> f64 {
        let _n = self.data.clone() * self.weight.clone() + self.bias.clone();
        _n.iter().fold(Array1::zeros(1), |acc, &x| acc + x)[0]
    }
}

// シグモイド関数.
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn main() {
    let mut n = Array1::<f64>::zeros(4);
    let mut w = Array1::<f64>::zeros(4);
    n[0] = 10.;
    w[0] = 2.;
    let _n = Neuron::new(10., n, w);
    println!("{:?}", _n.perceptron());
    println!("{}", sigmoid(_n.perceptron()));

    // シグモイド関数描画.
    let x: Vec<f64> = vec![-10., -9., -8., -7., -6., -5., -4., -3., -2., -1.,
                           0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    // haskellの[m..n]みたいな書き方はできないのだろうか？？？
    let y: Vec<f64> = x.iter().map(|i| sigmoid(*i)).collect();
    let mut fg = Figure::new();
    fg.axes2d().lines(&x, &y, &[]);
    fg.show();
}

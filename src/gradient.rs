extern crate nalgebra;

use std::boxed::Box;
use nalgebra::core::{DMatrix};
use neural;

// 勾配法.
pub struct GradientDescent {}

impl GradientDescent {
    /// 勾配法.
    ///
    /// ## 引数
    /// x: パラメータ.
    /// t: 教師データ.
    /// y: 出力結果.
    /// func: 勾配法を適用する損失関数.
    ///
    /// ## 戻り値
    //pub fn gradient<F: ::std::ops::Fn(&DMatrix<f64>, &DMatrix<f64>) -> f64>
    //               (d: &DMatrix<f64>, x: &mut DMatrix<f64>, f: F) -> Box<DMatrix<f64>> {
    pub fn gradient(obj: &neural::NeuralNW, x: &mut DMatrix<f64>) -> Box<DMatrix<f64>> {
        let mut result = DMatrix::<f64>::from_element(x.nrows(), x.ncols(), 0.);
        let _h = 1.0e-4;
        for i in 0 .. x.len() {
            // 1変数を変化させた場合の変化量を求める.
            let org = x[i];
            x[i] = org + _h;
            let _d1 = obj.callback(x);
            x[i] = org - _h;
            let _d2 = obj.callback(x);

            // 変化量を保存.
            result[i] = (_d1 - _d2) / (2. * _h);

            // 変化させたパラメータを元に戻す.
            x[i] = org;
        }
        return Box::new(result);
    }

    /// 微分関数.
    ///
    /// ## 引数
    /// x: 微分するXの値.
    /// func: 微分を求める関数.
    ///
    /// ## 戻り値
    /// f64: 微分値.
    pub fn numerical_diff<F: ::std::ops::Fn(f64) -> f64>(x: f64, func: F) -> f64 {
        let _d = 1.0e-5;
        (func(x + _d) - func(x - _d)) / (2. * _d)
    }
}

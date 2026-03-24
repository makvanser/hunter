use pyo3::prelude::*;
use std::collections::VecDeque;

#[pyclass]
pub struct KalmanFilter {
    state: [f64; 2],  // [beta, alpha]
    p: [[f64; 2]; 2], // Covariance 2x2
    q: [[f64; 2]; 2], // Process noise 2x2
    r: f64,           // Obs noise
    residuals: VecDeque<f64>,
    max_res_len: usize,
}

#[pymethods]
impl KalmanFilter {
    #[new]
    #[pyo3(signature = (process_noise=1e-5, observation_noise=1e-3))]
    pub fn new(process_noise: f64, observation_noise: f64) -> Self {
        Self {
            state: [1.0, 0.0],
            p: [[10.0, 0.0], [0.0, 10.0]],
            q: [[process_noise, 0.0], [0.0, process_noise]],
            r: observation_noise,
            residuals: VecDeque::with_capacity(100),
            max_res_len: 100,
        }
    }

    pub fn update(&mut self, price_alt: f64, price_btc: f64) -> (f64, f64, f64) {
        // 1. Prediction (Identity transition)
        self.p[0][0] += self.q[0][0];
        self.p[1][1] += self.q[1][1];

        // 2. Observation
        // log_alt = log_btc * beta + alpha
        let h = [price_btc, 1.0];
        let z = price_alt;

        let pred = h[0] * self.state[0] + h[1] * self.state[1];
        let residual = z - pred;

        // 3. Update
        // S = H * P * H^T + R
        let s = h[0] * (h[0] * self.p[0][0] + h[1] * self.p[1][0])
            + h[1] * (h[0] * self.p[0][1] + h[1] * self.p[1][1])
            + self.r;

        // Gain K = P * H^T / S
        let k0 = (self.p[0][0] * h[0] + self.p[0][1] * h[1]) / s;
        let k1 = (self.p[1][0] * h[0] + self.p[1][1] * h[1]) / s;

        // State update
        self.state[0] += k0 * residual;
        self.state[1] += k1 * residual;

        // P = (I - KH)P
        let kh00 = k0 * h[0];
        let kh01 = k0 * h[1];
        let kh10 = k1 * h[0];
        let kh11 = k1 * h[1];

        let p00_new = (1.0 - kh00) * self.p[0][0] - kh01 * self.p[1][0];
        let p01_new = (1.0 - kh00) * self.p[0][1] - kh01 * self.p[1][1];
        let p10_new = -kh10 * self.p[0][0] + (1.0 - kh11) * self.p[1][0];
        let p11_new = -kh10 * self.p[0][1] + (1.0 - kh11) * self.p[1][1];

        self.p = [[p00_new, p01_new], [p10_new, p11_new]];

        // Track residual
        if self.residuals.len() >= self.max_res_len {
            self.residuals.pop_front();
        }
        self.residuals.push_back(residual);

        (self.state[0], self.state[1], residual)
    }

    pub fn get_zscore(&self) -> f64 {
        if self.residuals.len() < 20 {
            return 0.0;
        }

        let mean: f64 = self.residuals.iter().sum::<f64>() / self.residuals.len() as f64;
        let variance: f64 = self
            .residuals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.residuals.len() as f64;

        let std = variance.sqrt();
        if std == 0.0 {
            return 0.0;
        }

        (self.residuals.back().unwrap_or(&0.0) - mean) / std
    }
}

#[pyclass]
pub struct OrderBook {
    bids: Vec<[f64; 2]>, // [price, qty]
    asks: Vec<[f64; 2]>,
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn new() -> Self {
        Self {
            bids: Vec::with_capacity(20),
            asks: Vec::with_capacity(20),
        }
    }

    pub fn update(&mut self, bids: Vec<[f64; 2]>, asks: Vec<[f64; 2]>) {
        self.bids = bids;
        self.asks = asks;
    }

    pub fn get_obi(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }

        let bid_vol: f64 = self.bids.iter().map(|level| level[1]).sum();
        let ask_vol: f64 = self.asks.iter().map(|level| level[1]).sum();

        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }

        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn hunter_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KalmanFilter>()?;
    m.add_class::<OrderBook>()?;
    Ok(())
}

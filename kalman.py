"""
Hunter V30 — Kalman Filter Engine
=================================
Provides real-time, dynamic estimation of the relationship (Beta/Alpha) 
between two assets (typically Altcoin and BTC).
"""

import numpy as np
from typing import Tuple

class KalmanFilter:
    """
    A 2D Kalman Filter to estimate y = beta * x + alpha.
    """
    def __init__(self, process_noise=1e-5, observation_noise=1e-3):
        # Initial state: [beta=1.0, alpha=0.0]
        # Most crypto is correlated 1:1 with BTC by default.
        self.state = np.array([1.0, 0.0])
        
        # Initial covariance matrix (high uncertainty)
        self.P = np.eye(2) * 10.0
        
        # Process noise covariance (Q) - how much we expect beta/alpha to drift
        self.Q = np.eye(2) * process_noise
        
        # Observation noise (R) - measurement error
        self.R = observation_noise
        
        # Residual tracking for Z-Score
        self.residuals = []
        self.max_res_len = 100

    def update(self, price_alt: float, price_btc: float) -> Tuple[float, float, float]:
        """
        Update the filter with new prices and return (beta, alpha, residual).
        """
        # 1. Prediction Step
        # x_hat = F * x (F is identity for random walk)
        # P = F * P * F.T + Q
        self.P += self.Q
        
        # 2. Measurement / Observation
        # y = price_alt
        # H = [price_btc, 1.0]
        H = np.array([price_btc, 1.0])
        
        # Prediction error (residual)
        # y_tilde = y - H * x_hat
        z = price_alt
        residual = z - np.dot(H, self.state)
        
        # 3. Update Step
        # S = H * P * H.T + R
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        
        # Kalman Gain: K = P * H.T * S^-1
        K = np.dot(self.P, H.T) / S
        
        # Update State: x = x + K * y_tilde
        self.state += K * residual
        
        # Update Covariance: P = (I - K * H) * P
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
        # Store residual for Z-score calculation
        self.residuals.append(residual)
        if len(self.residuals) > self.max_res_len:
            self.residuals.pop(0)
            
        beta, alpha = self.state
        return float(beta), float(alpha), float(residual)

    def get_zscore(self) -> float:
        """Calculate the Z-Score of the current residual."""
        if len(self.residuals) < 20:
            return 0.0
            
        res_arr = np.array(self.residuals)
        mean = np.mean(res_arr)
        std = np.std(res_arr)
        
        if std == 0:
            return 0.0
            
        return float((self.residuals[-1] - mean) / std)

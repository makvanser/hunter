"""
Hunter V25 — Telemetry & Observability
===========================================
Exports Prometheus metrics for real-time dashboarding via Grafana.
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import logging
import time

logger = logging.getLogger("hunter.telemetry")

# Prometheus Metrics
HUNTER_BALANCE = Gauge('hunter_balance_usd', 'Current real-time exchange balance')
HUNTER_OPEN_POSITIONS = Gauge('hunter_open_positions', 'Number of currently open positions')

HUNTER_TRADE_LATENCY = Histogram('hunter_trade_execution_latency_seconds', 'Time taken to execute an order via API')
HUNTER_TRADES_TOTAL = Counter('hunter_trades_total', 'Total number of trades executed', ['side', 'symbol'])
HUNTER_ERROR_COUNT = Counter('hunter_errors_total', 'Total system errors logged', ['type'])

HUNTER_ADX_VALUE = Gauge('hunter_adx_value', 'Real-time ADX value by symbol', ['symbol'])
HUNTER_DEEP_OBI = Gauge('hunter_deep_obi', 'Deep Orderbook Imbalance by symbol', ['symbol'])
HUNTER_ML_CONFIDENCE = Gauge('hunter_ml_confidence', 'ML Prediction Confidence for trade signals', ['symbol', 'action'])

class TelemetryManager:
    _server_started = False
    
    @classmethod
    def start_server(cls, port=8000):
        """Starts the Prometheus metrics HTTP server."""
        if not cls._server_started:
            try:
                start_http_server(port)
                logger.info(f"📊 Prometheus Metrics server listening on port {port}")
                cls._server_started = True
            except Exception as e:
                logger.error(f"❌ Failed to start Prometheus server: {e}")
                
    @staticmethod
    def track_latency():
        """Usage: with TelemetryManager.track_latency(): ..."""
        return HUNTER_TRADE_LATENCY.time()

    @staticmethod
    def set_balance(balance_usd: float):
        HUNTER_BALANCE.set(balance_usd)
        
    @staticmethod
    def set_open_positions(count: int):
        HUNTER_OPEN_POSITIONS.set(count)
        
    @staticmethod
    def inc_trade(side: str, symbol: str):
        HUNTER_TRADES_TOTAL.labels(side=side, symbol=symbol).inc()
        
    @staticmethod
    def inc_error(error_type: str):
        HUNTER_ERROR_COUNT.labels(type=error_type).inc()
        
    @staticmethod
    def set_adx(symbol: str, value: float):
        HUNTER_ADX_VALUE.labels(symbol=symbol).set(value)
        
    @staticmethod
    def set_deep_obi(symbol: str, value: float):
        HUNTER_DEEP_OBI.labels(symbol=symbol).set(value)
        
    @staticmethod
    def set_ml_confidence(symbol: str, action: str, confidence: float):
        HUNTER_ML_CONFIDENCE.labels(symbol=symbol, action=action).set(confidence)

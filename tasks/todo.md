# V25/V26 Institutional & Architectural Overhaul

Based on the Senior Architectural Review, we are moving away from pure Python/SQLite to an enterprise deployment model.

## Phase 1: High-Performance Database & Telemetry

- [x] Install Redis on the Ubuntu server.
- [x] Migrate `database.py`: Replace SQLite circuit breaker tracking and Positional locking with async Redis Cache (`redis-py`).
- [x] Create `telemetry.py`: Integrate `prometheus_client`.
- [x] Expose `/metrics` for order latency, balance tracking, and ML confidence.
- [x] Set up Grafana dashboard for live monitoring without print statements.

## Phase 2: ML Continuous Learning Pipeline

- [x] Update `ml.py`: Build `retrain_model()` to fetch the last 14 days of OHLCV data.
- [x] Implement Walk-Forward optimization logic.
- [x] Add a scheduler task in `main.py` (e.g., `apscheduler`) to retrain and override `ml_model.pkl` once a week.

## Phase 3: Zero-Latency Core Execution (NautilusTrader)

- [x] Initialize NautilusTrader core to replace Python's `asyncio` WebSocket feeds.
- [x] Port `provider.py`'s Depth20 and Kline streams to Rust-native handlers.
- [x] Benchmark order execution speed on Testnet: goal is sub-5 millisecond response time from tick to order placement API call.

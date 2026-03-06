# V25/V26 Institutional & Architectural Overhaul

Based on the Senior Architectural Review, we are moving away from pure Python/SQLite to an enterprise deployment model.

## Phase 1: High-Performance Database & Telemetry

- [ ] Install Redis on the Ubuntu server.
- [ ] Migrate `database.py`: Replace SQLite circuit breaker tracking and Positional locking with async Redis Cache (`redis-py`).
- [ ] Create `telemetry.py`: Integrate `prometheus_client`.
- [ ] Expose `/metrics` for order latency, balance tracking, and ML confidence.
- [ ] Set up Grafana dashboard for live monitoring without print statements.

## Phase 2: ML Continuous Learning Pipeline

- [ ] Update `ml.py`: Build `retrain_model()` to fetch the last 14 days of OHLCV data.
- [ ] Implement Walk-Forward optimization logic.
- [ ] Add a scheduler task in `main.py` (e.g., `apscheduler`) to retrain and override `ml_model.pkl` once a week.

## Phase 3: Zero-Latency Core Execution (NautilusTrader)

- [ ] Initialize NautilusTrader core to replace Python's `asyncio` WebSocket feeds.
- [ ] Port `provider.py`'s Depth20 and Kline streams to Rust-native handlers.
- [ ] Benchmark order execution speed on Testnet: goal is sub-5 millisecond response time from tick to order placement API call.

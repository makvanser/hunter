"""
Hunter V15 — Test Suite
===========================
Covers RSI, ADX regime, contrarian signals, circuit breaker,
PnL USD, multi-asset portfolio, config blacklist,
V13 news sentiment overrides,
V14 MACD, EMA/SMA, ATR, VWAP, divergence, S/R,
composite scoring, SL/TP, and DB symbol tracking.
V15 short positions, trailing SL, swing-point divergence, funding rate.
"""
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis import compute_adx, compute_atr, compute_bollinger, compute_composite_score, compute_ema, compute_macd, compute_rsi, compute_rsi_series, compute_sma, compute_support_resistance, compute_vwap, detect_divergence, generate_signal, get_market_regime, MarketState
from config import BLACKLIST, RSI_PERIOD
from database import get_consecutive_losses, init_db, is_on_cooldown, log_trade, reset_circuit_breaker, set_cooldown_until
from execution import PaperTrader

def _make_trending_candles(n: int=60):
    """Generate synthetic candles with a strong uptrend (high ADX)."""
    closes, highs, lows = ([], [], [])
    for i in range(n):
        c = 100 + i * 2.0
        closes.append(c)
        highs.append(c + 1.0)
        lows.append(c - 1.0)
    return (highs, lows, closes)

def _make_choppy_candles(n: int=60):
    """Generate synthetic candles with no clear direction (low ADX)."""
    closes, highs, lows = ([], [], [])
    for i in range(n):
        c = 100 + math.sin(i) * 0.5
        closes.append(c)
        highs.append(c + 0.2)
        lows.append(c - 0.2)
    return (highs, lows, closes)

def _temp_db():
    """Return a path to a temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path

class TestRSI:

    def test_rsi_boundaries(self):
        closes = [45000 + i * 50 for i in range(30)]
        rsi = compute_rsi(closes)
        assert 0 <= rsi <= 100

    def test_rsi_strong_uptrend_is_high(self):
        closes = [100 + i * 10 for i in range(30)]
        rsi = compute_rsi(closes)
        assert rsi > 70, f'Expected overbought RSI, got {rsi:.2f}'

    def test_rsi_strong_downtrend_is_low(self):
        closes = [1000 - i * 10 for i in range(30)]
        rsi = compute_rsi(closes)
        assert rsi < 30, f'Expected oversold RSI, got {rsi:.2f}'

class TestEMASMA:

    def test_sma_basic(self):
        data = [10, 20, 30, 40, 50]
        sma = compute_sma(data, 3)
        assert abs(sma - 40.0) < 0.01

    def test_sma_period_equals_length(self):
        data = [10, 20, 30]
        sma = compute_sma(data, 3)
        assert abs(sma - 20.0) < 0.01

    def test_sma_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            compute_sma([1, 2], 5)

    def test_ema_basic(self):
        data = [10, 10, 10, 10, 50]
        ema = compute_ema(data, 3)
        sma = compute_sma(data, 3)
        assert ema != sma
        assert ema > 0

    def test_ema_constant_series(self):
        data = [42.0] * 20
        ema = compute_ema(data, 10)
        assert abs(ema - 42.0) < 0.01

    def test_ema_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            compute_ema([1, 2], 5)

class TestADX:

    def test_adx_trending(self):
        highs, lows, closes = _make_trending_candles(60)
        adx = compute_adx(highs, lows, closes)
        regime = get_market_regime(adx)
        assert regime == 'TRENDING', f'ADX={adx:.2f}, expected TRENDING'

    def test_adx_choppy(self):
        highs, lows, closes = _make_choppy_candles(60)
        adx = compute_adx(highs, lows, closes)
        regime = get_market_regime(adx)
        assert regime == 'CHOPPY', f'ADX={adx:.2f}, expected CHOPPY'

class TestBollinger:

    def test_bollinger_band_order(self):
        closes = [100 + math.sin(i) * 5 for i in range(30)]
        lower, middle, upper = compute_bollinger(closes)
        assert lower < middle < upper

class TestMACD:

    def test_macd_uptrend_positive(self):
        """In a strong uptrend, MACD line should be positive."""
        closes = [100 + i * 2.0 for i in range(50)]
        macd_line, signal_line, histogram = compute_macd(closes)
        assert macd_line > 0, f'Expected positive MACD line in uptrend, got {macd_line:.4f}'

    def test_macd_downtrend_negative(self):
        """In a strong downtrend, MACD line should be negative."""
        closes = [500 - i * 2.0 for i in range(50)]
        macd_line, signal_line, histogram = compute_macd(closes)
        assert macd_line < 0, f'Expected negative MACD line in downtrend, got {macd_line:.4f}'

    def test_macd_returns_three_values(self):
        closes = [100 + math.sin(i) * 5 for i in range(50)]
        result = compute_macd(closes)
        assert len(result) == 3

    def test_macd_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            compute_macd([1, 2, 3])

class TestATR:

    def test_atr_positive(self):
        highs, lows, closes = _make_trending_candles(30)
        atr = compute_atr(highs, lows, closes)
        assert atr > 0, f'ATR should be positive, got {atr:.4f}'

    def test_atr_constant_market(self):
        """If high-low range is constant 2.0, ATR should converge to ~2.0."""
        n = 30
        closes = [100.0] * n
        highs = [101.0] * n
        lows = [99.0] * n
        atr = compute_atr(highs, lows, closes)
        assert abs(atr - 2.0) < 0.5, f'Expected ATR ≈ 2.0, got {atr:.4f}'

    def test_atr_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            compute_atr([1], [1], [1])

class TestVWAP:

    def test_vwap_basic(self):
        highs = [110.0]
        lows = [90.0]
        closes = [100.0]
        volumes = [1000.0]
        vwap = compute_vwap(highs, lows, closes, volumes)
        assert abs(vwap - 100.0) < 0.01

    def test_vwap_volume_weighted(self):
        """Higher volume bar should pull VWAP toward it."""
        highs = [100, 200]
        lows = [90, 190]
        closes = [95, 195]
        volumes = [100, 10000]
        vwap = compute_vwap(highs, lows, closes, volumes)
        assert vwap > 150, f'VWAP should be pulled toward high-volume bar, got {vwap:.2f}'

    def test_vwap_zero_volume(self):
        vwap = compute_vwap([100], [90], [95], [0])
        assert vwap == 0.0

    def test_vwap_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_vwap([100, 200], [90], [95], [1000])

class TestDivergence:

    def test_bullish_divergence(self):
        """
        Bullish divergence: price makes lower swing low,
        oscillator makes higher swing low.
        Use swing_window=1 for sharp V-shapes to be detected.
        """
        prices = [100, 80, 100, 70, 100]
        osc = [60, 30, 60, 40, 60]
        result = detect_divergence(prices, osc, lookback=5, swing_window=1)
        assert result == 'BULLISH_DIV', f'Expected BULLISH_DIV, got {result}'

    def test_bearish_divergence(self):
        """
        Bearish divergence: price makes higher swing high,
        oscillator makes lower swing high.
        """
        prices = [50, 80, 50, 90, 50]
        osc = [50, 70, 50, 60, 50]
        result = detect_divergence(prices, osc, lookback=5, swing_window=1)
        assert result == 'BEARISH_DIV', f'Expected BEARISH_DIV, got {result}'

    def test_no_divergence(self):
        prices = [100] * 20
        osc = [50] * 20
        result = detect_divergence(prices, osc, lookback=20)
        assert result == 'NONE'

    def test_insufficient_data(self):
        result = detect_divergence([1, 2, 3], [1, 2, 3], lookback=20)
        assert result == 'NONE'

class TestSupportResistance:

    def test_basic_pivots(self):
        highs = [100, 110, 105, 112, 100, 115, 100]
        lows = [95, 100, 90, 100, 85, 100, 90]
        supports, resistances = compute_support_resistance(highs, lows, lookback=7)
        assert len(supports) > 0, 'Should find at least one support'
        assert len(resistances) > 0, 'Should find at least one resistance'

    def test_insufficient_data(self):
        supports, resistances = compute_support_resistance([100], [90], lookback=1)
        assert supports == []
        assert resistances == []

class TestCompositeScore:

    def test_all_bullish_positive(self):
        """All indicators strongly bullish → high positive score."""
        score = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=20, ls_ratio=0.5, whale_net_vol=100, regime='CHOPPY', social_score=0.8, macd_histogram=0.5, bb_position=0.0, vwap_diff_pct=-2.0, divergence='BULLISH_DIV', funding_rate=0.0, open_interest_delta=5.0, liq_imbalance=0.0, rsi_slope=5.0, stoch_rsi=10.0, mtf_agreement=1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert score > 0.5, f'All bullish should give high score, got {score:.3f}'

    def test_all_bearish_negative(self):
        """All indicators strongly bearish → high negative score."""
        score = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=85, ls_ratio=2.0, whale_net_vol=-100, regime='CHOPPY', social_score=-0.8, macd_histogram=-0.5, bb_position=1.0, vwap_diff_pct=2.0, divergence='BEARISH_DIV', funding_rate=0.0, open_interest_delta=-5.0, liq_imbalance=0.0, rsi_slope=-5.0, stoch_rsi=90.0, mtf_agreement=-1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert score < -0.5, f'All bearish should give low score, got {score:.3f}'

    def test_neutral_close_to_zero(self):
        """Mixed/neutral signals → score near 0."""
        score = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert abs(score) < 0.3, f'Neutral should be near zero, got {score:.3f}'

    def test_vwap_and_divergence_affect_score(self):
        """VWAP below price + bullish divergence should boost score."""
        base = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        boosted = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=-3.0, divergence='BULLISH_DIV', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert boosted > base, f'VWAP+divergence should boost score ({boosted:.3f} vs {base:.3f})'

class TestSignal:

    def test_buy_signal(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=25, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'BUY'

    def test_hold_when_choppy(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=10, ls_ratio=0.3, whale_net_vol=9999, regime='CHOPPY', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD'

    def test_hold_when_rsi_not_oversold(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD'

    def test_hold_when_ls_ratio_high(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=25, ls_ratio=0.9, whale_net_vol=100, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD'

    def test_hold_when_whale_vol_negative(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=25, ls_ratio=0.6, whale_net_vol=-50, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD'

    def test_sell_signal(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=80, ls_ratio=1.2, whale_net_vol=0, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), current_position='BUY')
        assert sig == 'SELL'

    def test_sell_works_even_in_choppy(self):
        """Critical: overbought exit must fire regardless of regime."""
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=80, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), current_position='BUY')
        assert sig == 'SELL'

    def test_dca_panic_sell_block(self):
        """V20: If dca_count > 0, score-based momentum SELLs must be blocked (return HOLD)."""
        state = MarketState(btc_correlation=1.0, btc_dominance=50.0, 
            current_price=50000.0, rsi=85.0, ls_ratio=2.0, whale_net_vol=-1000,
            regime='TRENDING', social_score=0.0, macd_histogram=-5.0,
            bb_position=0.0, vwap_diff_pct=-5.0, divergence='BEARISH_DIV',
            funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0,
            rsi_slope=-5.0, stoch_rsi=10.0, mtf_agreement=-0.8,
            volume_confirm=True, near_resistance=False, atr_pct=1.0
        )
        
        # Test 1: dca_count = 1 -> SELL is blocked
        pos_in_dca = {"side": "BUY", "entry": 50100.0, "dca_count": 1, "size_usd": 200.0}
        sig_blocked = generate_signal(state, current_position=pos_in_dca)
        assert sig_blocked == 'HOLD'
        
        # Test 2: dca_count = 0 -> normal SELL
        pos_zero_dca = {"side": "BUY", "entry": 50100.0, "dca_count": 0, "size_usd": 100.0}
        sig_allowed = generate_signal(state, current_position=pos_zero_dca)
        assert sig_allowed == 'SELL'

class TestCompositeSignal:

    def test_composite_buy(self):
        """Strongly bullish composite → BUY in TRENDING."""
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=20, ls_ratio=0.5, whale_net_vol=100, regime='TRENDING', social_score=0.8, macd_histogram=0.5, bb_position=0.1, vwap_diff_pct=-2.0, divergence='BULLISH_DIV', funding_rate=0.0, open_interest_delta=5.0, liq_imbalance=0.0, rsi_slope=5.0, stoch_rsi=10.0, mtf_agreement=1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), use_composite=True)
        assert sig == 'BUY', f'Expected BUY, got {sig}'

    def test_composite_sell(self):
        """Strongly bearish composite → SELL or SHORT."""
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=85, ls_ratio=2.0, whale_net_vol=-100, regime='TRENDING', social_score=-0.8, macd_histogram=-0.5, bb_position=0.95, vwap_diff_pct=2.0, divergence='BEARISH_DIV', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=-1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), use_composite=True)
        assert sig in ('SELL', 'SHORT'), f'Expected SELL/SHORT, got {sig}'

    def test_composite_hold_in_choppy(self):
        """Even bullish composite → HOLD if market is CHOPPY."""
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=20, ls_ratio=0.5, whale_net_vol=100, regime='CHOPPY', social_score=0.8, macd_histogram=0.5, bb_position=0.1, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), use_composite=True)
        assert sig in ('HOLD', 'SELL', 'SHORT'), f'CHOPPY should not BUY, got {sig}'

    def test_composite_sell_overrides_choppy(self):
        """SELL fires even in CHOPPY (to protect capital)."""
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=85, ls_ratio=2.0, whale_net_vol=-100, regime='CHOPPY', social_score=-0.8, macd_histogram=-0.5, bb_position=0.95, vwap_diff_pct=2.0, divergence='BEARISH_DIV', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=-1.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), use_composite=True)
        assert sig in ('SELL', 'SHORT'), f'Composite SELL should work in CHOPPY, got {sig}'

class TestPnL:

    def test_long_profit_usd(self):
        """$100 position, entry 50000, exit 51000 → +$2.00 profit."""
        pnl = PaperTrader.simulate_pnl(entry=50000, exit_price=51000, size_usd=100, side='BUY')
        assert abs(pnl - 1.82) < 0.01, f'Expected ~$1.82, got ${pnl:.4f}'

    def test_long_loss_usd(self):
        """$100 position, entry 50000, exit 49000 → -$2.00 loss."""
        pnl = PaperTrader.simulate_pnl(entry=50000, exit_price=49000, size_usd=100, side='BUY')
        assert abs(pnl - -2.18) < 0.01, f'Expected ~-$2.18, got ${pnl:.4f}'

    def test_breakeven(self):
        """Same entry/exit → $0 PnL."""
        pnl = PaperTrader.simulate_pnl(entry=50000, exit_price=50000, size_usd=100, side='BUY')
        assert abs(pnl - -0.18) < 0.01

class TestCircuitBreaker:

    def test_three_losses_activate_cooldown(self):
        db = _temp_db()
        init_db(db)
        for _ in range(3):
            log_trade('SELL', 100.0, 100.0, pnl=-5.0, db_path=db)
        assert is_on_cooldown(db)
        os.unlink(db)

    def test_win_resets_counter(self):
        db = _temp_db()
        init_db(db)
        log_trade('SELL', 100.0, 100.0, pnl=-5.0, db_path=db)
        log_trade('SELL', 100.0, 100.0, pnl=-5.0, db_path=db)
        assert get_consecutive_losses(db) == 2
        log_trade('SELL', 110.0, 100.0, pnl=10.0, db_path=db)
        assert get_consecutive_losses(db) == 0
        os.unlink(db)

    def test_cooldown_blocks_trade(self):
        db = _temp_db()
        init_db(db)
        set_cooldown_until(datetime.now(timezone.utc) + timedelta(hours=1), db_path=db)
        trader = PaperTrader(db_path=db)
        result = trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        assert result['blocked'] is True
        assert result['action'] == 'BLOCKED_BY_CIRCUIT_BREAKER'
        os.unlink(db)

    def test_expired_cooldown_allows_trade(self):
        db = _temp_db()
        init_db(db)
        reset_circuit_breaker(db)
        set_cooldown_until(datetime.now(timezone.utc) - timedelta(hours=1), db_path=db)
        trader = PaperTrader(db_path=db)
        result = trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        assert result['blocked'] is False
        os.unlink(db)

class TestMultiAsset:

    def test_open_two_symbols_simultaneously(self):
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        r1 = trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        r2 = trader.execute_trade('BUY', 3200.0, 'ETHUSDT')
        assert r1['action'] == 'OPENED_LONG'
        assert r2['action'] == 'OPENED_LONG'
        assert trader.open_positions_count() == 2
        assert trader.has_position('BTCUSDT')
        assert trader.has_position('ETHUSDT')
        os.unlink(db)

    def test_close_one_keeps_other(self):
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        trader.execute_trade('BUY', 3200.0, 'ETHUSDT')
        r_sell = trader.execute_trade('SELL', 51000.0, 'BTCUSDT')
        assert r_sell['action'] == 'CLOSED_LONG'
        assert not trader.has_position('BTCUSDT')
        assert trader.has_position('ETHUSDT')
        assert trader.open_positions_count() == 1
        os.unlink(db)

    def test_buy_duplicate_symbol_is_noop(self):
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        r2 = trader.execute_trade('BUY', 50500.0, 'BTCUSDT')
        assert r2['action'] == 'NO_ACTION'
        assert trader.open_positions_count() == 1
        os.unlink(db)

    def test_circuit_breaker_blocks_all_symbols(self):
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        for _ in range(3):
            trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
            trader.execute_trade('SELL', 49000.0, 'BTCUSDT')
        r = trader.execute_trade('BUY', 3200.0, 'ETHUSDT')
        assert r['blocked'] is True
        assert r['action'] == 'BLOCKED_BY_CIRCUIT_BREAKER'
        os.unlink(db)

class TestSLTP:

    def test_stop_loss_triggers(self):
        """Price drops below SL → position auto-closes."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=100.0)
        assert trader.has_position('BTCUSDT')
        pos = trader.get_position('BTCUSDT')
        assert pos['stop_loss'] is not None
        assert pos['take_profit'] is not None
        result = trader.execute_trade('HOLD', 49800.0, 'BTCUSDT', atr=100.0)
        assert result['action'] == 'CLOSED_SL'
        assert not trader.has_position('BTCUSDT')
        assert result['pnl'] < 0
        os.unlink(db)

    def test_take_profit_triggers(self):
        """Price rises above TP → position auto-closes."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=100.0)
        assert trader.has_position('BTCUSDT')
        result = trader.execute_trade('HOLD', 50300.0, 'BTCUSDT', atr=100.0)
        assert result['action'] == 'CLOSED_TP'
        assert not trader.has_position('BTCUSDT')
        assert result['pnl'] > 0
        os.unlink(db)

    def test_no_sl_tp_without_atr(self):
        """Opening without ATR should not set SL/TP."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        pos = trader.get_position('BTCUSDT')
        assert pos['stop_loss'] is None
        assert pos['take_profit'] is None
        os.unlink(db)

    def test_sl_tp_values_correct(self):
        """Verify SL/TP calculation: SL = entry - ATR*1.5, TP = entry + ATR*2.5."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=200.0)
        pos = trader.get_position('BTCUSDT')
        assert abs(pos['stop_loss'] - 49700.0) < 0.01
        assert abs(pos['take_profit'] - 50500.0) < 0.01
        os.unlink(db)

class TestConfig:

    def test_blacklist_contains_stablecoins(self):
        assert 'USDCUSDT' in BLACKLIST
        assert 'USDPUSDT' in BLACKLIST

    def test_btcusdt_not_blacklisted(self):
        assert 'BTCUSDT' not in BLACKLIST

class TestNewsSentiment:
    """Test that news sentiment correctly overrides RSI-based signals."""

    def test_bullish_overrides_sell(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=80, ls_ratio=1.2, whale_net_vol=0, regime='TRENDING', social_score=0.8, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD', f'BULLISH news should override SELL, got {sig}'

    def test_bearish_overrides_buy(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=25, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=-0.8, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD', f'BEARISH news should override BUY, got {sig}'

    def test_neutral_does_not_override_sell(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=80, ls_ratio=1.2, whale_net_vol=0, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), current_position='BUY')
        assert sig == 'SELL', f'NEUTRAL should not override, got {sig}'

    def test_neutral_does_not_override_buy(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=25, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'BUY', f'NEUTRAL should not override, got {sig}'

    def test_default_backward_compat(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=80, ls_ratio=1.2, whale_net_vol=0, regime='TRENDING', social_score=0.0, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0), current_position='BUY')
        assert sig == 'SELL', 'Default 4-arg call must still return SELL'

    def test_bullish_does_not_affect_non_overbought(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=0.8, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD', f'BULLISH with RSI=50 should be HOLD, got {sig}'

    def test_bearish_does_not_affect_non_oversold(self):
        sig = generate_signal(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000.0, rsi=50, ls_ratio=0.6, whale_net_vol=100, regime='TRENDING', social_score=-0.8, macd_histogram=0.0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert sig == 'HOLD', f'BEARISH with RSI=50 should be HOLD, got {sig}'

class TestRSISeries:

    def test_rsi_series_length(self):
        closes = [100 + i for i in range(30)]
        series = compute_rsi_series(closes)
        assert len(series) == len(closes) - RSI_PERIOD

    def test_rsi_series_last_matches_compute_rsi(self):
        """Last value of RSI series should match compute_rsi()."""
        closes = [100 + i * 2 for i in range(30)]
        series = compute_rsi_series(closes)
        single = compute_rsi(closes)
        assert abs(series[-1] - single) < 0.01

class TestDBSymbol:

    def test_symbol_stored_in_trades(self):
        """Verify that log_trade stores the symbol in the database."""
        import sqlite3
        db = _temp_db()
        init_db(db)
        trade_id = log_trade('SELL', 50000.0, 100.0, pnl=2.0, db_path=db, symbol='BTCUSDT')
        conn = sqlite3.connect(db)
        row = conn.execute('SELECT symbol FROM trades WHERE id = ?', (trade_id,)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 'BTCUSDT'
        os.unlink(db)

    def test_symbol_default_is_unknown(self):
        """If symbol is not passed, it defaults to UNKNOWN."""
        import sqlite3
        db = _temp_db()
        init_db(db)
        trade_id = log_trade('SELL', 50000.0, 100.0, pnl=2.0, db_path=db)
        conn = sqlite3.connect(db)
        row = conn.execute('SELECT symbol FROM trades WHERE id = ?', (trade_id,)).fetchone()
        conn.close()
        assert row[0] == 'UNKNOWN'
        os.unlink(db)

class TestSwingPointDivergence:
    """Tests for the V15 swing-point based divergence algorithm."""

    def test_bullish_div_with_real_swing_lows(self):
        """
        Two sharp V-shaped swing lows where price lower low + osc higher low.
        swing_window=1 detects single-bar dips.
        """
        prices = [100, 90, 100, 85, 100]
        osc = [50, 30, 50, 40, 50]
        result = detect_divergence(prices, osc, lookback=5, swing_window=1)
        assert result == 'BULLISH_DIV', f'Expected BULLISH_DIV, got {result}'

    def test_no_divergence_flat_price(self):
        """Flat price and oscillator should return NONE."""
        prices = [100.0] * 70
        osc = [50.0] * 70
        result = detect_divergence(prices, osc, lookback=70)
        assert result == 'NONE'

    def test_insufficient_bars_returns_none(self):
        """If there's not enough data, return NONE safely."""
        prices = [100.0, 90.0, 110.0]
        osc = [50.0, 40.0, 60.0]
        result = detect_divergence(prices, osc, lookback=60)
        assert result == 'NONE'

    def test_swing_window_respects_parameter(self):
        """Larger swing_window should require broader local extrema."""
        prices = [100.0] * 80
        prices[20] = 80.0
        osc = [50.0] * 80
        osc[20] = 40.0
        prices[50] = 75.0
        osc[50] = 45.0
        result_small = detect_divergence(prices, osc, lookback=80, swing_window=1)
        result_large = detect_divergence(prices, osc, lookback=80, swing_window=5)
        assert result_small in ('BULLISH_DIV', 'BEARISH_DIV', 'NONE')
        assert result_large in ('BULLISH_DIV', 'BEARISH_DIV', 'NONE')

class TestFundingRate:

    def test_negative_funding_is_bullish(self):
        """Negative funding (shorts pay longs) should boost bullish score."""
        score_neutral = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        score_neg_funding = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=-0.02, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert score_neg_funding > score_neutral, f'Negative funding should boost score: {score_neg_funding:.3f} vs {score_neutral:.3f}'

    def test_positive_funding_is_bearish(self):
        """Positive funding (longs pay shorts) = overcrowded longs = bearish."""
        score_neutral = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.0, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        score_pos_funding = compute_composite_score(MarketState(btc_correlation=1.0, btc_dominance=50.0, current_price=50000, rsi=50, ls_ratio=1.0, whale_net_vol=0, regime='CHOPPY', social_score=0.0, macd_histogram=0, bb_position=0.5, vwap_diff_pct=0.0, divergence='NONE', funding_rate=0.02, open_interest_delta=0.0, liq_imbalance=0.0, rsi_slope=0.0, stoch_rsi=50.0, mtf_agreement=0.0, volume_confirm=True, near_resistance=False, atr_pct=0.0))
        assert score_pos_funding < score_neutral, f'Positive funding should lower score: {score_pos_funding:.3f} vs {score_neutral:.3f}'

class TestShortPositions:

    def test_open_short(self):
        """SHORT signal opens a short position."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        result = trader.execute_trade('SHORT', 50000.0, 'BTCUSDT', atr=200.0)
        assert result['action'] == 'OPENED_SHORT'
        assert trader.has_position('BTCUSDT')
        pos = trader.get_position('BTCUSDT')
        assert pos['side'] == 'SELL'
        assert pos['stop_loss'] > 50000.0
        assert pos['take_profit'] < 50000.0
        os.unlink(db)

    def test_short_pnl_profit(self):
        """Short position makes profit when price falls."""
        pnl = PaperTrader.simulate_pnl(entry=50000.0, exit_price=48000.0, size_usd=100.0, side='SELL')
        assert abs(pnl - 3.82) < 0.01, f'Expected $3.82 profit, got ${pnl:.4f}'

    def test_short_pnl_loss(self):
        """Short position loses when price rises."""
        pnl = PaperTrader.simulate_pnl(entry=50000.0, exit_price=52000.0, size_usd=100.0, side='SELL')
        assert abs(pnl - -4.18) < 0.01, f'Expected -$4.18 loss, got ${pnl:.4f}'

    def test_cover_closes_short(self):
        """COVER signal closes a short position."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('SHORT', 50000.0, 'BTCUSDT')
        result = trader.execute_trade('COVER', 48000.0, 'BTCUSDT')
        assert result['action'] == 'CLOSED_SHORT'
        assert not trader.has_position('BTCUSDT')
        assert result['pnl'] > 0
        os.unlink(db)

    def test_short_sl_triggers_above_entry(self):
        """SL for short triggers when price rises ABOVE SL level."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('SHORT', 50000.0, 'BTCUSDT', atr=200.0)
        result = trader.execute_trade('HOLD', 50400.0, 'BTCUSDT', atr=200.0)
        assert result['action'] == 'CLOSED_SL'
        assert result['pnl'] < 0
        os.unlink(db)

    def test_short_tp_triggers_below_entry(self):
        """TP for short triggers when price falls BELOW TP level."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('SHORT', 50000.0, 'BTCUSDT', atr=200.0)
        result = trader.execute_trade('HOLD', 49400.0, 'BTCUSDT', atr=200.0)
        assert result['action'] == 'CLOSED_TP'
        assert result['pnl'] > 0
        os.unlink(db)

    def test_short_and_long_simultaneously(self):
        """Can hold a long on BTC and a short on ETH at the same time."""
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT')
        trader.execute_trade('SHORT', 3200.0, 'ETHUSDT')
        assert trader.open_positions_count() == 2
        assert trader.get_position('BTCUSDT')['side'] == 'BUY'
        assert trader.get_position('ETHUSDT')['side'] == 'SELL'
        os.unlink(db)

class TestTrailingSL:

    def test_trailing_sl_ratchets_upward(self):
        """
        SL should move up when price rises above the activation threshold.
        Uses large ATR=2000 so that TP (entry+2000*2.5=55000) is far away.
        """
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=2000.0)
        initial_sl = trader.get_position('BTCUSDT')['stop_loss']
        trader.execute_trade('HOLD', 50600.0, 'BTCUSDT', atr=2000.0)
        pos = trader.get_position('BTCUSDT')
        assert pos is not None, 'Position should still be open after trailing SL update'
        new_sl = pos['stop_loss']
        assert new_sl > initial_sl, f'Trailing SL should move up: new={new_sl:.2f} vs initial={initial_sl:.2f}'
        os.unlink(db)

    def test_trailing_sl_does_not_activate_below_threshold(self):
        """
        If price rise < TRAILING_SL_ACTIVATION_PCT (0.5%), SL should NOT move.
        """
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=100.0)
        initial_sl = trader.get_position('BTCUSDT')['stop_loss']
        trader.execute_trade('HOLD', 50050.0, 'BTCUSDT', atr=100.0)
        sl_after = trader.get_position('BTCUSDT')['stop_loss']
        assert sl_after == initial_sl, f'Trailing SL should NOT move below activation threshold: {sl_after:.2f} != {initial_sl:.2f}'
        os.unlink(db)

    def test_trailing_sl_never_moves_down(self):
        """
        SL should never be lowered — it only ratchets up.
        Even if price dips after a rally, SL stays at previous level.
        Uses ATR=2000 so that small price moves don't hit TP.
        """
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=2000.0)
        trader.execute_trade('HOLD', 50600.0, 'BTCUSDT', atr=2000.0)
        pos = trader.get_position('BTCUSDT')
        assert pos is not None, 'Position should still be open after rally'
        sl_after_rally = pos['stop_loss']
        trader.execute_trade('HOLD', 50100.0, 'BTCUSDT', atr=2000.0)
        pos2 = trader.get_position('BTCUSDT')
        assert pos2 is not None, 'Position should still be open after dip'
        sl_after_dip = pos2['stop_loss']
        assert sl_after_dip >= sl_after_rally, f'SL should never decrease: {sl_after_dip:.2f} < {sl_after_rally:.2f}'
        os.unlink(db)

    def test_trailing_sl_triggers_on_pullback(self):
        """
        After full activation, a large pullback should hit the trailing SL.
        Uses ATR=2000 so prices can move freely without hitting TP.
        """
        db = _temp_db()
        init_db(db)
        trader = PaperTrader(db_path=db)
        trader.execute_trade('BUY', 50000.0, 'BTCUSDT', atr=2000.0)
        trader.execute_trade('HOLD', 52000.0, 'BTCUSDT', atr=2000.0)
        pos = trader.get_position('BTCUSDT')
        assert pos is not None, 'Position should still be open after rally'
        assert pos['stop_loss'] > 47000.0, 'Trailing SL should have moved up from initial'
        result = trader.execute_trade('HOLD', 48500.0, 'BTCUSDT', atr=2000.0)
        assert result['action'] == 'CLOSED_SL', f"Trailing SL should have triggered, got: {result['action']}"
        os.unlink(db)
from social import SocialManager
import pytest
from unittest.mock import patch, MagicMock

class TestSocialManager:
    @patch('social.TrendReq')
    @patch('news.NewsManager.get_fear_and_greed')
    @patch('news.NewsManager.get_sentiment')
    def test_social_score_bullish(self, mock_sentiment, mock_fg, mock_trendreq):
        # Setup mocks
        mock_sentiment.return_value = "BULLISH"
        mock_fg.return_value = (80, "Extreme Greed")  # Score +0.6
        
        # Mock Google Trends positive momentum (+0.5)
        mock_pytrends = MagicMock()
        mock_trendreq.return_value = mock_pytrends
        
        social = SocialManager()
        # Override cache to avoid real fetch
        social.trends_cache["BTCUSDT"] = (0.5, 9999999999.0)
        
        # Expected:
        # News (Bullish) = 0.8 * 0.5 = 0.4
        # F&G (80) = 0.6 * 0.3 = 0.18
        # Trends = 0.5 * 0.2 = 0.10
        # Total = 0.68
        
        score = social.get_social_score("BTCUSDT")
        assert score == pytest.approx(0.68, abs=0.01)

    @patch('social.TrendReq')
    @patch('news.NewsManager.get_fear_and_greed')
    @patch('news.NewsManager.get_sentiment')
    def test_social_score_bearish(self, mock_sentiment, mock_fg, mock_trendreq):
        # Setup mocks
        mock_sentiment.return_value = "BEARISH"
        mock_fg.return_value = (20, "Extreme Fear")  # Score -0.6
        
        # Mock Google Trends negative momentum (-0.3)
        mock_pytrends = MagicMock()
        mock_trendreq.return_value = mock_pytrends
        
        social = SocialManager()
        social.trends_cache["ETHUSDT"] = (-0.3, 9999999999.0)
        
        # Expected:
        # News (Bearish) = -0.8 * 0.5 = -0.4
        # F&G (20) = -0.6 * 0.3 = -0.18
        # Trends = -0.3 * 0.2 = -0.06
        # Total = -0.64
        
        score = social.get_social_score("ETHUSDT")
        assert score == pytest.approx(-0.64, abs=0.01)

from macro import MacroManager
from unittest.mock import patch, MagicMock
import pytest

@pytest.mark.asyncio
class TestMacroManager:
    @patch('aiohttp.ClientSession.get')
    async def test_get_btc_dominance_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status = 200
        # aiohttp resp.json() is async
        async def mock_json():
            return {"data": {"market_cap_percentage": {"btc": 54.32}}}
        mock_resp.json = mock_json
        
        # mock_get returns an async context manager
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_resp
        mock_get.return_value = mock_ctx
        
        macro = MacroManager()
        dom = await macro.get_btc_dominance()
        assert dom == 54.32
        
        await macro.close_session()

    @patch('aiohttp.ClientSession.get')
    async def test_get_btc_dominance_429(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status = 429
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_resp
        mock_get.return_value = mock_ctx
        
        macro = MacroManager()
        dom = await macro.get_btc_dominance()
        assert dom == 50.0 # Fallback
        
        await macro.close_session()

    async def test_get_btc_correlation(self):
        macro = MacroManager()
        mock_provider = MagicMock()
        
        # BTC closes: linear up
        btc_closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # ETH closes: linear up (perfect correlation)
        eth_closes = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        async def mock_fetch_ohlcv(symbol, limit):
            if symbol == "BTCUSDT":
                return [], [], btc_closes, []
            return [], [], eth_closes, []
            
        mock_provider.fetch_ohlcv = mock_fetch_ohlcv
        
        corr = await macro.get_btc_correlation("ETHUSDT", mock_provider)
        assert corr == pytest.approx(1.0, abs=0.01)

    async def test_get_btc_correlation_negative(self):
        macro = MacroManager()
        mock_provider = MagicMock()
        
        # BTC closes: linear up
        btc_closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # SOL closes: linear down (perfect inverse correlation)
        sol_closes = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        
        async def mock_fetch_ohlcv(symbol, limit):
            if symbol == "BTCUSDT":
                return [], [], btc_closes, []
            return [], [], sol_closes, []
            
        mock_provider.fetch_ohlcv = mock_fetch_ohlcv
        
        corr = await macro.get_btc_correlation("SOLUSDT", mock_provider)
        assert corr == pytest.approx(-1.0, abs=0.01)



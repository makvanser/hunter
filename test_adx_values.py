import asyncio
from config import TIMEFRAME, KLINE_LIMIT, TOP_PAIRS_COUNT
from provider import BinanceProvider
from analysis import compute_adx

async def main():
    async with BinanceProvider() as provider:
        top_pairs = await provider.scan_top_pairs(10)
        for sym in top_pairs:
            highs, lows, closes, volumes = await provider.fetch_ohlcv(sym, TIMEFRAME, 50) # 50 limits might be enough
            
            # Fetch default KLINE_LIMIT to see exactly what main.py sees
            highs, lows, closes, volumes = await provider.fetch_ohlcv(sym, TIMEFRAME, KLINE_LIMIT)
            adx = compute_adx(highs, lows, closes, 14)
            print(f"{sym} ADX: {adx:.2f}")

if __name__ == "__main__":
    asyncio.run(main())

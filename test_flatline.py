import asyncio
from provider import BinanceProvider

async def main():
    async with BinanceProvider() as provider:
        for sym in ["TANSSIUSDT", "NEIROETHUSDT", "OMNIUSDT", "ALPHAUSDT"]:
            h, l, c, v = await provider.fetch_ohlcv(sym, "1h", 20)
            print(f"{sym} last 10 volumes: {v[-10:]}")
            print(f"{sym} last 10 closes: {c[-10:]}")

if __name__ == "__main__":
    asyncio.run(main())

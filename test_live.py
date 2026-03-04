import asyncio
import logging
import pprint
from live_execution import LiveTrader

logging.basicConfig(level=logging.INFO)

async def test_trade():
    trader = LiveTrader()
    
    print("\n--- OPENING LONG ---")
    # Using dummy current price of 90000 - the API will execute at MARKET anyway
    res = await trader.execute_trade("BUY", 90000.0, "BTCUSDT", atr=500.0)
    pprint.pprint(res)
    
    print("\n--- WAITING 3 SECONDS ---")
    await asyncio.sleep(3)
    
    print("\n--- CLOSING LONG ---")
    res2 = await trader.execute_trade("SELL", 90000.0, "BTCUSDT", atr=500.0)
    pprint.pprint(res2)

if __name__ == '__main__':
    asyncio.run(test_trade())

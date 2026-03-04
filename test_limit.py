import asyncio
import logging
from live_execution import LiveTrader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("test")

async def main():
    trader = LiveTrader()
    await trader.sync_balance()
    
    symbol = "BTCUSDT"
    logger.info("Executing limit BUY on %s. Balance: $%.2f", symbol, trader.balance)
    
    # We will fetch current price manually via symbol ticker
    import aiohttp
    from config import BASE_URL
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/fapi/v1/ticker/price?symbol={symbol}") as r:
            data = await r.json()
            current_price = float(data['price'])
    
    # Place Limit BUY POST_ONLY slightly below current price (to act as maker)
    # If the price is 69000, we'll try 68500 to ensure we don't immediately fill
    # but the polling logic should wait. Actually, let's place it at current_price - 100
    target_price = current_price - 100.0
    quantity = 0.05
    
    logger.info("Current Price: %.2f. Attempting Limit BUY @ %.2f", current_price, target_price)
    
    avg_price = await trader.open_limit_order(symbol, "BUY", quantity, target_price)
    
    logger.info("Result avg_price: %.4f", avg_price)

if __name__ == "__main__":
    asyncio.run(main())

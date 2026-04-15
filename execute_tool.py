import sys
import asyncio
import logging

try:
    from live_execution import LiveTrader
    from provider import BinanceProvider
except ImportError:
    print("Error: Must be run from inside the hunter skill directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("hunter.executor_tool")

async def main():
    if len(sys.argv) < 4:
        print("Usage: python3 execute_tool.py <SYMBOL> <BUY|SHORT|SELL|COVER> <USD_SIZE>")
        print("Example: python3 execute_tool.py BTCUSDT BUY 15.0")
        sys.exit(1)
        
    symbol = sys.argv[1].upper()
    action = sys.argv[2].upper()
    
    try:
        size_usd = float(sys.argv[3])
    except ValueError:
        print("Error: USD_SIZE must be a number.")
        sys.exit(1)

    if action not in ["BUY", "SHORT", "SELL", "COVER"]:
        print(f"Error: Invalid action '{action}'. Use BUY, SHORT, SELL, or COVER.")
        sys.exit(1)

    logger.info("🤖 OpenClaw Executor Agent invoked trade: %s %s (Size: $%s)", action, symbol, size_usd)

    # Fetch current price from provider
    async with BinanceProvider() as provider:
        bbo = await provider.get_bbo(symbol)
        
        if not bbo or bbo['bid'] == 0 or bbo['ask'] == 0:
            print(f"Error: Could not fetch orderbook for {symbol}. Is the symbol correct?")
            sys.exit(1)
            
        # Determine execution price (Limit Maker constraint)
        # To ensure maker fee rebate: we bid at Best Bid, and ask at Best Ask.
        current_price = bbo['bid'] if action in ["BUY", "COVER"] else bbo['ask']
        
        logger.info("📊 Current BBO for %s -> Bid: %.4f, Ask: %.4f. Target Price: %.4f", 
                    symbol, bbo['bid'], bbo['ask'], current_price)

        trader = LiveTrader()
        
        # OpenLimitOrder logic automatically handles sizing from USD to Token amount
        print(f"Sending LIMIT {action} order for {symbol} at {current_price} (Size: ${size_usd})...")
        success = await trader.open_limit_order(symbol, action, size_usd, current_price)
        
        if success:
            print(f"✅ Trade {action} {symbol} successfully routed to Binance!")
        else:
            print(f"❌ Failed to route trade. Check bot_live.log for details.")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import aiohttp
import time
import hmac
import hashlib
import sqlite3
import os
from config import BASE_URL, API_KEY, API_SECRET

async def fetch_positions(session):
    url = f"{BASE_URL}/fapi/v2/positionRisk"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    
    headers = {'X-MBX-APIKEY': API_KEY}
    async with session.get(f"{url}?{query_string}&signature={signature}", headers=headers) as resp:
        return await resp.json()

async def cancel_all_orders(session, symbol):
    url = f"{BASE_URL}/fapi/v1/allOpenOrders"
    timestamp = int(time.time() * 1000)
    query_string = f"symbol={symbol}&timestamp={timestamp}"
    signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    
    headers = {'X-MBX-APIKEY': API_KEY}
    async with session.delete(f"{url}?{query_string}&signature={signature}", headers=headers) as resp:
        return await resp.json()

async def close_position(session, symbol, position_amt):
    amt = float(position_amt)
    if amt == 0:
        return

    side = "SELL" if amt > 0 else "BUY"
    quantity = abs(amt)

    url = f"{BASE_URL}/fapi/v1/order"
    timestamp = int(time.time() * 1000)
    
    payload = {
        'symbol': symbol,
        'side': side,
        'type': 'MARKET',
        'quantity': str(quantity),
        'reduceOnly': 'true',
        'timestamp': timestamp
    }
    
    query_string = '&'.join([f"{k}={v}" for k, v in payload.items()])
    signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    payload['signature'] = signature
    
    headers = {'X-MBX-APIKEY': API_KEY}
    async with session.post(url, headers=headers, data=payload) as resp:
        print(f"[{symbol}] Closed position {amt}. Status: {resp.status}")
        print(await resp.json())

async def main():
    print("🧹 Очистка дашборда Binance (Фьючерсы)...")
    async with aiohttp.ClientSession() as session:
        # 1. Получаем все позиции
        positions = await fetch_positions(session)
        
        if isinstance(positions, dict) and 'code' in positions:
            print(f"⚠️ Ошибка API: {positions}")
            return
            
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        if not open_positions:
            print("✅ Открытых позиций нет.")
        else:
            for p in open_positions:
                symbol = p['symbol']
                amt = p['positionAmt']
                print(f"Найдена позиция {symbol} (Размер: {amt}). Отменяем ордера и закрываем...")
                
                # Отменяем открытые лимитки для этого символа
                await cancel_all_orders(session, symbol)
                # Закрываем по маркету
                await close_position(session, symbol, amt)
                
    # 2. Очистка локальной базы данных
    db_paths = ["hunter_live.db", "backtest.db"]
    for db in db_paths:
        if os.path.exists(db):
            os.remove(db)
            print(f"🗑️ Удалена локальная база: {db}")
    
    print("🎉 Все старые сделки закрыты! Дашборд чист.")

if __name__ == "__main__":
    asyncio.run(main())

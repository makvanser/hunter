import asyncio
import aiohttp
import time
import hmac
import hashlib
from config import API_KEY, API_SECRET, BASE_URL

async def close_position():
    async with aiohttp.ClientSession() as session:
        payload = {
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': '0.004',
            'timestamp': int(time.time() * 1000)
        }
        query_string = '&'.join([f"{k}={v}" for k, v in payload.items()])
        sig = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        payload['signature'] = sig
        url = f"{BASE_URL}/fapi/v1/order"
        headers = {'X-MBX-APIKEY': API_KEY}
        async with session.post(url, headers=headers, data=payload) as response:
            print("Status:", response.status)
            print(await response.json())

if __name__ == '__main__':
    asyncio.run(close_position())

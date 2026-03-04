import asyncio
import aiohttp
import time
import hmac
import hashlib
from config import API_KEY, API_SECRET, BASE_URL

async def get_positions():
    async with aiohttp.ClientSession() as session:
        payload = {'timestamp': int(time.time() * 1000)}
        query_string = '&'.join([f"{k}={v}" for k, v in payload.items()])
        sig = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        payload['signature'] = sig
        url = f"{BASE_URL}/fapi/v2/positionRisk"
        headers = {'X-MBX-APIKEY': API_KEY}
        async with session.get(url, headers=headers, params=payload) as response:
            data = await response.json()
            for p in data:
                if float(p['positionAmt']) != 0:
                    print(f"OPEN POSITION: {p['symbol']} | Amt: {p['positionAmt']} | Entry: {p['entryPrice']} | UnPnL: {p['unRealizedProfit']}")

if __name__ == '__main__':
    asyncio.run(get_positions())

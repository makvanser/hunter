import asyncio, aiohttp, hmac, hashlib, time, json
from config import API_KEY, API_SECRET, BASE_URL

async def check():
    ts = int(time.time()*1000)
    qs = f'timestamp={ts}'
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f'{BASE_URL}/fapi/v2/balance?{qs}&signature={sig}'
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={'X-MBX-APIKEY': API_KEY}) as r:
            data = await r.json()
            for a in data:
                if float(a.get("balance", 0)) != 0:
                    print(f"{a['asset']}: balance={a['balance']}, crossWallet={a.get('crossWalletBalance','?')}, available={a.get('availableBalance','?')}")

asyncio.run(check())

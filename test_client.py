"""
test_client.py
Simple test client to start a job and print WebSocket messages.
Requirements: pip install websockets requests
Usage: python test_client.py
"""
import asyncio
import requests
import websockets

API_BASE = "http://localhost:8000"

async def run():
    # Start a demo run
    resp = requests.post(f"{API_BASE}/run", json={"case": "Demo", "demo": True})
    data = resp.json()
    job_id = data["job_id"]
    ws_path = data["ws"]
    print("Started job:", job_id, "WS:", ws_path)

    scheme = "ws"
    host = "localhost:8000"
    ws_url = f"{scheme}://{host}{ws_path}"
    async with websockets.connect(ws_url) as ws:
        try:
            async for msg in ws:
                print(msg)
        except websockets.ConnectionClosed:
            print("WebSocket closed")

if __name__ == "__main__":
    asyncio.run(run())
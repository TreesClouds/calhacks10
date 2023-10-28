import asyncio
import json

from hume import HumeStreamClient, StreamSocket
from hume.models.config import FacemeshConfig

HUME_KEY = 'A6wdNigFGRQjP7q3616ICffWKxVwpOTTGB60f7IoFnFZAj1R'

async def main():
    client = HumeStreamClient(HUME_KEY)
    config = FacemeshConfig()
    async with client.connect([config]) as socket:
        fm = open("example.txt", "r")
        result = await socket.send_facemesh(fm.read())
        emotions = sorted(result['face']['predictions'][0]['emotions'], key=lambda x: x['score'], reverse=True)
        top_emotions = emotions[:5]
        print([i['name'] for i in top_emotions])

asyncio.run(main())

import asyncio
from engine import SovereignBBEngine

if __name__ == "__main__":
    target = "https://example.com"
    engine = SovereignBBEngine(concurrency=40)
    asyncio.run(engine.run(target))

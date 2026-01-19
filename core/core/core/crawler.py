import aiohttp
from urllib.parse import urlparse, parse_qs

async def crawl(url: str):
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    vectors = []
    for p in params:
        vectors.append({
            "url": url,
            "parameter": p,
            "type": "query"
        })

    return vectors

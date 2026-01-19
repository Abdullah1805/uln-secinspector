import aiohttp
import time

PAYLOADS = {
    "XSS": "<script>alert(1)</script>",
    "SQLi": "' OR '1'='1'--"
}

async def analyze(vector):
    url = vector["url"]
    param = vector["parameter"]

    async with aiohttp.ClientSession() as session:
        for k, payload in PAYLOADS.items():
            test_url = url.replace(
                f"{param}=",
                f"{param}={payload}"
            )

            start = time.time()
            async with session.get(test_url) as r:
                text = await r.text()
                delay = time.time() - start

                if payload in text or delay > 5:
                    return {
                        "title": f"{k} detected",
                        "parameter": param,
                        "type": k,
                        "confidence": round(min(delay / 5, 1.0), 2),
                        "evidence": text[:500]
                    }

    return None

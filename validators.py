import aiohttp, asyncio, time

class ValidatorEngine:

    async def validate(self, f):
        if f["type"] == "PARAM_TEST":
            return await self._test_sqli(f), "Time delay detected"

        return False, None

    async def _test_sqli(self, f):
        payload = "1' AND SLEEP(5)--"
        params = {f["param"]: payload}

        async with aiohttp.ClientSession() as session:
            start = time.time()
            try:
                async with session.get(f["url"], params=params, timeout=10) as r:
                    pass
            except:
                return False

        return (time.time() - start) > 4.5

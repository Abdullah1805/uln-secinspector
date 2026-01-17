import aiohttp
import asyncio
import time

class ValidatorEngine:
    async def validate(self, f):
        if f["type"] == "PARAM_TEST":
            is_vulnerable = await self._test_sqli(f)
            return is_vulnerable, "Time delay detected (Sleep 5s)"
        return False, None

    async def _test_sqli(self, f):
        payload = "1' AND SLEEP(5)--"
        params = {f["param"]: payload}
        
        async with aiohttp.ClientSession() as session:
            start = time.time()
            try:
                # نستخدم timeout أكبر من وقت الـ sleep للتأكد من التقاط الاستجابة
                async with session.get(f["url"], params=params, timeout=10) as r:
                    await r.text()
            except:
                return False
            
            duration = time.time() - start
            return duration > 4.5

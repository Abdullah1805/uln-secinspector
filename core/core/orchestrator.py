from core.crawler import crawl
from core.analyzer import analyze

class F16Orchestrator:
    def __init__(self, target: str):
        self.target = target

    async def run(self):
        vectors = await crawl(self.target)
        findings = []

        for v in vectors:
            res = await analyze(v)
            if res:
                findings.append(res)

        return findings

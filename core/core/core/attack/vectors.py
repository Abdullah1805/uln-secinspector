# ============================================================
# Injection Vector Extraction (Research‑Grade)
# ============================================================

from urllib.parse import urlparse, parse_qs, urljoin
from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class InjectionPoint:
    url: str
    method: str          # GET / POST
    parameter: str
    location: str        # query / body / dom
    input_type: str      # text / hidden / email / js / unknown


class VectorExtractor:
    """
    لا Regex أعمى.
    تحليل DOM حقيقي.
    """

    @staticmethod
    async def extract(page, base_url: str) -> List[InjectionPoint]:
        vectors: List[InjectionPoint] = []

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        # --------------------
        # 1) URL Query Params
        # --------------------
        parsed = urlparse(base_url)
        qs = parse_qs(parsed.query)
        for param in qs:
            vectors.append(
                InjectionPoint(
                    url=base_url,
                    method="GET",
                    parameter=param,
                    location="query",
                    input_type="text"
                )
            )

        # --------------------
        # 2) Forms
        # --------------------
        for form in soup.find_all("form"):
            action = form.get("action") or base_url
            method = form.get("method", "get").upper()
            target_url = urljoin(base_url, action)

            for inp in form.find_all("input"):
                name = inp.get("name")
                if not name:
                    continue

                vectors.append(
                    InjectionPoint(
                        url=target_url,
                        method=method,
                        parameter=name,
                        location="body",
                        input_type=inp.get("type", "text")
                    )
                )

        return vectors

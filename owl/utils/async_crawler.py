import asyncio, aiohttp, re, hashlib, os
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from readability import Document               # pip install readability-lxml
from typing import List, Set, Dict

class AsyncCrawler:
    def __init__(
        self,
        max_depth: int = 1,
        limit: int = 20,
        concurrency: int = 8,
        cache_dir: str = "tmp/",
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ):
        self.max_depth, self.limit = max_depth, limit
        self.sem = asyncio.Semaphore(concurrency)
        self.seen: Set[str] = set()
        self.include_patterns, self.exclude_patterns = include_patterns, exclude_patterns
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        async with self.sem:
            try:
                # 单次请求最多等待 20 秒。
                async with session.get(url, timeout=20) as r:
                    # 只返回包含 "text/html" 的响应体，其它类型一律忽略
                    if "text/html" in r.headers.get("Content-Type", ""):
                        return await r.text()
            except Exception:
                return ""

    '''
    包含模式：如果提供了 include_patterns，URL 必须至少匹配一个正则，否则跳过。
    排除模式：如果提供了 exclude_patterns，URL 只要匹配任意一个就排除。
    '''
    def _match(self, url: str) -> bool:
        p = lambda pats, u: any(re.search(pat, u) for pat in pats or [])
        return (not self.include_patterns or p(self.include_patterns, url)) and \
               (not self.exclude_patterns or not p(self.exclude_patterns, url))

    async def crawl(self, url: str) -> List[Dict]:
        queue = [(url, 0)]
        results = []

        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as sess:
            while queue and len(results) < self.limit:
                cur, depth = queue.pop(0)
                if cur in self.seen or depth > self.max_depth or not self._match(cur):
                    continue
                self.seen.add(cur)
                html = await self.fetch(sess, cur)
                if not html:
                    continue

                soup = BeautifulSoup(html, "html.parser")
                results.append({"url": cur, "html": html, "soup": soup})

                # enqueue new links
                if depth < self.max_depth:
                    for a in soup.find_all("a", href=True):
                        nxt = urljoin(cur, a["href"])
                        if urlparse(nxt).scheme in ("http", "https"):
                            queue.append((nxt, depth + 1))
        return results

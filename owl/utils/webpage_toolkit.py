import os

import aiohttp
from camel.toolkits import BaseToolkit, ImageAnalysisToolkit, FunctionTool
from camel.utils import retry_on_error

from owl.utils.async_crawler import AsyncCrawler
from owl.utils.page_extractor import PageExtractor


class WebPageToolkit(BaseToolkit):
    def __init__(self, model=None, cache_dir="tmp/"):
        self.crawler = AsyncCrawler(cache_dir=cache_dir)
        self.image_tool = ImageAnalysisToolkit(model=model)
        self.extractor = PageExtractor(img_dir=os.path.join(cache_dir, "imgs"),
                                       img_toolkit=self.image_tool)

    @retry_on_error()
    async def crawl_and_extract(self, url: str,
                                max_depth: int = 1,
                                limit: int = 20) -> str:
        pages = await self.crawler.crawl(url)
        async with aiohttp.ClientSession() as sess:
            md_chunks = [await self.extractor.parse_page(p, sess) for p in pages]
        return "\n\n---\n\n".join(md_chunks)

    def get_tools(self):
        return [FunctionTool(self.crawl_and_extract)]

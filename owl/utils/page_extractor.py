import hashlib
import os
from urllib.parse import urljoin
from xml.dom.minidom import Document

import aiofiles, mimetypes, pathlib, asyncio
from bs4 import BeautifulSoup


class PageExtractor:
    IMG_ATTRS = ["src", "data-src", "data-original", "data-lazy-src"]

    def __init__(self, img_dir: str, img_toolkit):
        self.img_dir = img_dir
        self.img_toolkit = img_toolkit
        os.makedirs(img_dir, exist_ok=True)

    async def _download_img(self, session, url) -> str:
        # 1) 依据 URL 生成本地文件名（md5 + 后缀）
        # 2) 如果已下载，直接复用
        # 3) 否则异步 GET，确认 Content-Type 是 image，再保存到本地
        # 4) 返回本地路径，失败返回空串
        fname = hashlib.md5(url.encode()).hexdigest() + pathlib.Path(url).suffix
        path = os.path.join(self.img_dir, fname)
        if os.path.exists(path):
            return path
        try:
            async with session.get(url) as r:
                if r.status == 200 and r.headers.get("Content-Type", "").startswith("image"):
                    async with aiofiles.open(path, "wb") as f:
                        await f.write(await r.read())
                    return path
        except: ...
        return ""

    async def parse_page(self, page, session) -> str:
        # 1) 用 readability.Document 提取“主要内容” HTML 片段
        # 2) 用 BeautifulSoup 解析出 <img> 标签，收集可能的图片链接
        # 3) 并行下载前 N 张图片（通过 _download_img）
        # 4) 对每个下载成功的图片，调用 img_toolkit.ask_question_about_image：
        #       prompt = "请识别图片中的文字并用一句话描述关键信息。"
        #    将非空 caption 拼成 Markdown 格式：`![img](本地路径)\n*caption*`
        # 5) 用 BeautifulSoup 再次将正文 HTML 转成纯文本
        # 6) 最终返回：
        #       ### 原始页面 URL
        #       正文纯文本
        #       图片 Markdown + caption 列表
        doc = Document(page["html"])
        main_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(main_html, "html.parser")

        # ── 1. 收集图片链接 ──
        img_tasks = []
        for tag in soup.find_all("img"):
            for attr in self.IMG_ATTRS:
                if tag.get(attr):
                    img_url = urljoin(page["url"], tag[attr])
                    img_tasks.append(asyncio.create_task(self._download_img(session, img_url)))
                    break

        local_paths = [p for p in await asyncio.gather(*img_tasks) if p]

        # ── 2. 图片→caption ──
        captions = []
        for pth in local_paths[:20]:
            prompt = "请识别图片中的文字并用一句话描述关键信息。"
            cap = self.img_toolkit.ask_question_about_image(pth, prompt)
            if cap and cap.lower() != "none":
                captions.append(f"![img]({pth})\n*{cap}*")

        # ── 3. 返回 Markdown ──
        text_md = BeautifulSoup(main_html, "html.parser").get_text("\n")
        return f"### {page['url']}\n\n{text_md}\n\n" + "\n\n".join(captions)

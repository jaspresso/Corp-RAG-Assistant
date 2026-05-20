"""
단일 URL 을 Crawl4AI 로 가져와 Markdown 텍스트를 LlamaIndex ``Document`` 로 만든다.

``AsyncWebCrawler`` 는 async context manager 로 브라우저/세션 자원을 정리한다.
네트워크·차단·타임아웃 시 ``arun`` 이 실패하거나 ``markdown`` 이 비어 있을 수 있으니
호출 측에서 필요하면 검증·재시도를 추가한다.
"""
from crawl4ai import AsyncWebCrawler
from llama_index.core import Document


async def crawl_website(url: str) -> Document:
    """
    주어진 ``url`` 페이지를 크롤링해 본문을 Markdown 으로 추출한다.

    Args:
        url: HTTP(S) 페이지 주소.

    Returns:
        ``text`` 에 마크다운, ``metadata`` 에 ``url`` 과 ``source: web`` 이 들어간 Document.
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        markdown_text = result.markdown or ""
        print(f"✅ [크롤링 성공] {url}에서 {len(markdown_text)}자 추출 완료")

        return Document(
            text=markdown_text,
            metadata={"url": url, "source": "web"},
        )

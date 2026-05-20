"""
``web_crawler.crawl_website`` 단독 스모크 테스트.

프로젝트 루트에서 실행:
    ``python test_crawler.py``

네트워크가 필요하며, 대상 사이트 정책에 따라 크롤이 차단될 수 있다.
"""
import asyncio

from src.ingestion.web_crawler import crawl_website


async def test() -> None:
    """임의 URL 을 크롤해 추출 텍스트 앞부분만 출력한다."""
    url = "https://www.google.com"
    doc = await crawl_website(url)
    print("--- 추출된 데이터 일부 ---")
    print(doc.text[:500])


if __name__ == "__main__":
    asyncio.run(test())

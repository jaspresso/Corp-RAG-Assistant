"""
RAG 인덱싱(적재) 파이프라인.

역할 요약:
    1. PDF 디렉터리와(선택) 웹 URL에서 LlamaIndex Document 를 모은다.
    2. 문장 단위 청킹 → HuggingFace 임베딩(BGE-M3)을 거친 뒤
    3. Qdrant 의 `corporate_knowledge` 컬렉션에 벡터를 저장한다.

실행 방법:
    프로젝트 루트에서 `python -m src.ingestion.pipeline` 또는 이 파일을 직접 실행.
    (Qdrant 가 `http://localhost:6333` 에 떠 있어야 하며, 첫 실행 시 컬렉션/인덱스가 생성됨.)

주의:
    `llm_service.py` 의 질의 엔진과 동일한 임베딩 모델·컬렉션명을 써야 검색 품질이 맞는다.
"""
import asyncio
import os
import sys

# 스크립트를 어디서 실행하든 `src.*` 임포트가 되도록 프로젝트 루트를 sys.path 에 넣는다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import qdrant_client
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.ingestion.pdf_parser import parse_pdf_to_documents
from src.ingestion.web_crawler import crawl_website


def get_ingestion_pipeline() -> IngestionPipeline:
    """
    Qdrant 에 연결된 IngestionPipeline 을 구성해 반환한다.

    변환 순서:
        Document 리스트
            → SentenceSplitter: 긴 텍스트를 고정 길이(토큰 근사) 청크로 나눔
            → HuggingFaceEmbedding: 각 청크를 벡터로 변환
            → vector_store: 벡터와 메타데이터를 Qdrant 에 upsert

    Returns:
        실행 준비가 된 LlamaIndex IngestionPipeline 인스턴스.
    """
    # Qdrant HTTP API 클라이언트. 로컬 기본 포트는 6333.
    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    # LlamaIndex 가 Qdrant 컬렉션에 읽기/쓰기할 때 쓰는 어댑터.
    # collection_name 은 질의 쪽(`llm_service.py`)과 반드시 일치해야 한다.
    # enable_hybrid=False: 희소(sparse)+밀집(dense) 하이브리드 색인/검색을 끔.
    #   환경에 따라 하이브리드 관련 설정 불일치가 나기 쉬워, 단순 벡터 적재만 사용.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="corporate_knowledge",
        enable_hybrid=False,
    )

    # 서빙 시와 동일한 임베딩을 써야 "같은 공간"에서 유사도 검색이 된다.
    # BGE-M3 는 다국어·긴 문맥에 강한 벡터 모델(쿼리용은 별도 프리픽스가 있는 경우도 있으나
    #   기본 HuggingFaceEmbedding 사용 시 ingest/query 동일 모델이면 대부분 일관됨).
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # transformations 리스트 순서대로 각 Document 처리 후 최종 결과가 vector_store 로 전달된다.
    pipeline = IngestionPipeline(
        transformations=[
            # chunk_size: 한 청크의 대략적 크기(문자/토큰 정책은 라이브러리 구현 따름).
            # chunk_overlap: 인접 청크 간 겹침 — 문장이 경계에서 잘리는 것을 완화.
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            embed_model,
        ],
        vector_store=vector_store,
    )
    return pipeline


def run_full_pipeline(documents: list) -> None:
    """
    주어진 Document 리스트를 파이프라인에 넣고, 청킹·임베딩 후 Qdrant 에 저장한다.

    Args:
        documents: LlamaIndex `Document` 인스턴스의 리스트.
            각 Document 는 `text` 와 `metadata`(출처 파일명, URL 등)를 가질 수 있다.

    Side effects:
        Qdrant 컬렉션에 노드(청크) 단위로 벡터가 추가된다.
        반환값은 없고, 처리 개수는 로그/프린트로 확인한다.
    """
    pipeline = get_ingestion_pipeline()
    print(f"🚀 총 {len(documents)}개의 문서를 파이프라인에 투입합니다.")
    # pipeline.run 은 내부적으로 split → embed → persist 를 수행하고, 생성된 노드 리스트를 반환한다.
    nodes = pipeline.run(documents=documents)
    print(f"✅ 처리 완료: {len(nodes)}개의 노드가 Qdrant에 저장되었습니다.")


async def main() -> None:
    """
    스크립트 직접 실행 시: PDF 폴더 + 예시 웹 URL 을 한꺼번에 수집해 적재한다.

    웹 크롤러가 async 이므로 엔트리포인트를 async main + asyncio.run 으로 둔다.
    """
    # 상대 경로는 **실행 시 현재 작업 디렉터리(CWD)** 기준이다.
    # 보통 저장소 루트에서 실행하도록 맞추면 `./data/pdfs` 가 의도한 폴더를 가리킨다.
    PDF_DIR = "./data/pdfs"
    # 실제 사내 정책 페이지 등으로 교체. Crawl4AI 로 HTML → 마크다운 텍스트 추출.
    TARGET_URL = "https://example.com/company-policy"

    all_docs = []

    # --- 1) 로컬 PDF: Docling 으로 마크다운 변환 후 Document 리스트로 합침 ---
    print("📥 PDF 수집 중...")
    all_docs.extend(parse_pdf_to_documents(PDF_DIR))

    # --- 2) 웹: 단일 URL 을 크롤해 Document 하나(또는 빈 텍스트)를 추가 ---
    print(f"🌐 웹 데이터 수집 중: {TARGET_URL}")
    all_docs.append(await crawl_website(TARGET_URL))

    # --- 3) 수집 결과가 있으면 전체를 한 번의 파이프라인 실행으로 Qdrant 에 반영 ---
    if all_docs:
        run_full_pipeline(all_docs)


if __name__ == "__main__":
    asyncio.run(main())

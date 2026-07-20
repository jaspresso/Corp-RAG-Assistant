"""
RAG 인덱싱(적재) 파이프라인.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import qdrant_client
from qdrant_client.http import models as rest
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.ingestion.pdf_parser import parse_pdf_to_documents

COLLECTION_NAME = "corporate_knowledge_v11"


def get_ingestion_pipeline() -> IngestionPipeline:
    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    # Qdrant 컬렉션이 없으면 bge-m3(1024차원) 구조로 직접 강제 생성
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"⚙️ Qdrant 컬렉션 `{COLLECTION_NAME}` 이 존재하지 않아 새롭게 생성합니다...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=1024,
                distance=rest.Distance.COSINE,
            ),
        )
        print(f"✅ `{COLLECTION_NAME}` 컬렉션 생성 완료!")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=False,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=2048, chunk_overlap=100),
            embed_model,
        ],
        vector_store=vector_store,
    )
    return pipeline


def main() -> None:
    PDF_DIR = "./data/pdfs"
    print("📥 PDF 수집 시작...")
    
    all_docs = parse_pdf_to_documents(PDF_DIR)

    if not all_docs:
        print("❌ 파싱된 Document가 없습니다. 적재를 중단합니다.")
        return

    print(f"🚀 총 {len(all_docs)}개의 페이지 문서를 Qdrant 파이프라인에 적재합니다...")
    pipeline = get_ingestion_pipeline()
    nodes = pipeline.run(documents=all_docs)
    print(f"🎉 성공! 총 {len(nodes)}개의 노드가 `{COLLECTION_NAME}` 컬렉션에 정밀 저장되었습니다.")


if __name__ == "__main__":
    main()
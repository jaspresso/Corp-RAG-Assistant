"""
RAG 질의 엔진 구성 (LlamaIndex + Qdrant + Ollama + HuggingFace 임베딩).

``ingestion/pipeline.py`` 에서 적재할 때와 **동일한** 임베딩 모델·Qdrant 컬렉션명을 써야
벡터 공간이 일치하고 검색 품질이 나온다.

외부 의존:
    - Qdrant: ``http://localhost:6333`` (컬렉션 ``corporate_knowledge``)
    - Ollama: 로컬에서 ``llama3`` 모델 서빙
    - HuggingFace: ``BAAI/bge-m3`` (최초 실행 시 모델 다운로드 가능)
"""
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client


def get_chat_engine():
    """
    Qdrant 에 저장된 벡터 인덱스를 로드해, 질의응답용 QueryEngine 을 반환한다.

    LlamaIndex ``Settings`` 는 전역 기본값이다. 이 프로젝트에서는 매 호출마다
    임베딩/LLM 을 덮어쓰므로, 다른 모듈과 섞일 때는 호출 순서에 유의한다.

    Returns:
        ``index.as_query_engine()`` — ``.query(str)`` 로 동기 질의 가능한 엔진.

    Raises:
        연결/모델 오류는 하위 레이어에서 예외로 전파될 수 있다 (Qdrant 다운, Ollama 미기동 등).
    """
    # 인제션 파이프라인과 동일 임베딩 — 다르면 쿼리 벡터와 문서 벡터가 맞지 않는다.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # 로컬 Ollama HTTP API. 모델명은 ``ollama pull llama3`` 등으로 사전 준비 필요.
    Settings.llm = Ollama(model="llama3")

    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="corporate_knowledge",
    )

    # 기존 컬렉션만 읽어 인덱스 구성 (문서 추가는 ingestion 쪽).
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine()

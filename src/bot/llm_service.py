"""
RAG 질의 엔진 구성 (LlamaIndex + Qdrant + Ollama + HuggingFace 임베딩).
"""
from functools import lru_cache

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# 💡 OCR 노이즈를 감안하여 문맥을 넓게 해석하도록 개선한 프롬프트
KOREAN_TEXT_QA_PROMPT = PromptTemplate(
    """당신은 사내 및 교원 복무 규정을 안내하는 AI 도우미입니다.
제공된 참고 문서(OCR 스캔본으로 일부 문자 오탈자가 존재할 수 있음)를 바탕으로 질문에 답변하세요.

[답변 작성 수칙]
1. 문서 내용 기반 답변:
   - 문서에 수치나 규정이 부분적으로라도 포함되어 있다면, OCR 오탈자를 감안하여 문맥을 정제한 후 답변하세요.
   - 7년차, 3년차 등 특정 근속년수 질문이 들어오면 문서 내의 재직기간 구간 표(예: 6년 이상, 3~4년 등)와 대조하여 유추 계산해서 대답하세요.
   - 단, 문서에 정말로 관련된 언급(급여, 호봉표 등)이 아예 없는 경우에만 "해당 정보가 명시되어 있지 않습니다"라고 답변하세요.

2. 출산/유휴가 및 수당 관련 복합 질문:
   - 질문에 여러 조건(근무년수, 출산휴가, 임신 등)이 섞여 있더라도, 문서에서 찾을 수 있는 출산휴가 일수/임신 순산 휴가 규정을 핵심 위주로 정리해 주세요.

---------------------
[참고 문서 내용]
{context_str}
---------------------

질문: {query_str}
한국어 답변: """
)


@lru_cache(maxsize=1)
def get_chat_engine():
    # 임베딩 모델 로드
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # LLM 설정 (Llama 3.1)
    Settings.llm = Ollama(
        model="llama3.1",
        request_timeout=120.0,
        additional_kwargs={
            "system": "당신은 사내 규정 안내 AI입니다. OCR 텍스트의 오탈자를 감안하여 참고 문서에서 최선의 정답을 찾아 설명하세요.",
            "temperature": 0.0
        },
    )

    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    # src/bot/llm_service.py 내부 QdrantVectorStore 부분 수정
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="corporate_knowledge_v12" # v12로 변경
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine(
        text_qa_template=KOREAN_TEXT_QA_PROMPT,
        response_mode=ResponseMode.COMPACT,
        similarity_top_k=6,  # 💡 검색 청크를 6개로 늘려 정보 누락 차단!
    )
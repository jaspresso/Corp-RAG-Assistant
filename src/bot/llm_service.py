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

# 💡 [핵심] 재직기간별 연가일수 구간 매칭 및 사례 문제 문단 간섭 방지 지침 추가
KOREAN_TEXT_QA_PROMPT = PromptTemplate(
    """당신은 사내 규정 안내 전문 AI 챗봇입니다.
아래 제공된 참고 문서(OCR 스캔 텍스트)의 규정을 정확히 분석하여 질문에 대답하세요.

[답변 작성 가이드라인]
1. 재직기간별 연가일수 질문:
   - '재직기간에 따른 연가일수가 21일인 교사가 있다'와 같은 [사례/예시 문제] 문장의 숫자를 정답으로 사용하지 마세요.
   - 반드시 문서 내부의 [재직기간별 연가 일수 표 규정]을 기준으로 대답하세요.
   - 규정 구간 기준:
     * 1년 이상 3년 미만: 15일
     * 3년 이상 4년 미만: 16일
     * 4년 이상 5년 미만: 17일
     * 5년 이상 6년 미만: 20일 (재직기간 5년인 경우 해당)
     * 6년 이상: 21일
2. 미성년 자녀 유급휴가 질문:
   - 한부모/장애인 가산일을 제외한 '기본 자녀수별 유급휴가' (1명=2일, 2명=3일, 3명=4일) 기준을 적용하세요.
3. 교원 근무시간 질문:
   - 교원 특수 근무시간인 09:00 ~ 17:00를 정확히 인용하세요.
4. 영어를 사용하지 말고 생각 과정 없이 한국어 결과 문장만 깔끔하게 대답하세요.

---------------------
{context_str}
---------------------

질문: {query_str}
한국어 답변: """
)


@lru_cache(maxsize=1)
def get_chat_engine():
    # 임베딩 모델 로드
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # LLM 설정
    Settings.llm = Ollama(
        model="llama3",
        request_timeout=120.0,
        additional_kwargs={
            "system": "당신은 사내 규정 안내 AI 챗봇입니다. 사례 예시가 아닌 규정 표의 내용에 기반하여 정확하게 한국어로만 답변합니다.",
            "temperature": 0.0
        },
    )

    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="corporate_knowledge_v11"
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine(
        text_qa_template=KOREAN_TEXT_QA_PROMPT,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        similarity_top_k=4,
    )
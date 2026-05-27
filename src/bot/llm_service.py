"""
RAG 질의 엔진 구성 (LlamaIndex + Qdrant + Ollama + HuggingFace 임베딩).

``ingestion/pipeline.py`` 에서 적재할 때와 **동일한** 임베딩 모델·Qdrant 컬렉션명을 써야
벡터 공간이 일치하고 검색 품질이 나온다.

외부 의존:
    - Qdrant: ``http://localhost:6333`` (컬렉션 ``corporate_knowledge``)
    - Ollama: 로컬에서 ``llama3`` 모델 서빙
    - HuggingFace: ``BAAI/bge-m3`` (최초 실행 시 모델 다운로드 가능)
"""
from functools import lru_cache

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# LlamaIndex 기본 COMPACT 모드는 청크 2개 이상일 때 "refine" 단계를 거친다.
# 기본 refine 프롬프트가 영어라, 2번째 청크 처리 후 답이 영어로 바뀌는 경우가 많다.
# (질문 1~2는 우연히 한국어로 남고, 3번째부터 검색 문맥이 길/영어이면 영어로 고정되는 패턴)
KOREAN_TEXT_QA_PROMPT = PromptTemplate(
    """다음은 참고 문맥입니다.
---------------------
{context_str}
---------------------
위 문맥과 질문만을 근거로 답하세요. 사전 지식만으로 추측하지 마세요.
반드시 전체 답변을 한국어로만 작성하세요.
영어(다른 언어 포함)로 답하면 오답 처리됩니다.
필요한 경우 문서의 고유명사/약어는 원문 형태로 유지할 수 있지만,
문장 전체는 한국어로만 작성하세요.

질문: {query_str}
답변: """
)

KOREAN_REFINE_PROMPT = PromptTemplate(
    """원래 질문: {query_str}
기존 답변: {existing_answer}
아래 추가 문맥을 참고해, 필요할 때만 기존 답을 더 좋게 다듬으세요.
문맥이 도움이 되지 않으면 기존 답변을 그대로 두세요.
반드시 전체 답변을 한국어로만 유지/재작성하세요.
영어(다른 언어 포함)로 바꾸면 오답 처리됩니다.
고유명사/약어만 원문 형태를 유지할 수 있습니다.

추가 문맥:
------------
{context_msg}
------------
개선된 답변: """
)


@lru_cache(maxsize=1)
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
    # 속도 최적화:
    # - 이 함수는 Chainlit 세션 시작 시/서버 기동 시 한 번만 준비하면 된다.
    # - lru_cache 로 임베딩 모델 로딩, 인덱스 로딩, Qdrant 연결을 재사용한다.

    # 인제션 파이프라인과 동일 임베딩 — 다르면 쿼리 벡터와 문서 벡터가 맞지 않는다.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # 로컬 Ollama HTTP API. 모델명은 ``ollama pull llama3`` 등으로 사전 준비 필요.
    #
    # 한국어 응답을 “기본값”으로 강제하기 위해 Ollama의 system 프롬프트를 지정한다.
    # (사용자가 영어로 요청하거나 번역을 원하면 그 요구를 우선한다.)
    korean_system_prompt = (
        "당신은 사내 지식 기반 RAG 챗봇입니다. 기본적으로 한국어로 간결하고 정확하게 답변하세요. "
        "사용자가 영어로 답해달라고 하거나 번역을 요청하면 그에 따르세요. "
        "근거가 부족하면 추측하지 말고 필요한 추가 정보를 요청하세요."
    )
    Settings.llm = Ollama(
        model="llama3",
        additional_kwargs={"system": korean_system_prompt},
    )

    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="corporate_knowledge",
    )

    # 기존 컬렉션만 읽어 인덱스 구성 (문서 추가는 ingestion 쪽).
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine(
        text_qa_template=KOREAN_TEXT_QA_PROMPT,
        # COMPACT/REFINE 모드는 상황에 따라 refine 단계(2차 LLM 호출)가 발생합니다.
        # 사용자 요구사항(언어 일관성/속도)을 위해 단일 호출 중심으로 변경합니다.
        # SIMPLE_SUMMARIZE: 모든 청크를 합친 뒤 LLM 1회로 답변 생성
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        # 체감 속도 개선:
        # - top_k=1로 줄이면 COMPACT 모드에서 refine 단계가 사실상 사라져
        #   LLM 추가 호출(정제)이 줄어드는 경우가 많다.
        similarity_top_k=1,
    )

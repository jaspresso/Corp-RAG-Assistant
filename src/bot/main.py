"""
FastAPI 로 RAG 챗봇을 HTTP API 로 노출하는 엔트리포인트.

실행 예:
    ``uvicorn src.bot.main:app --reload`` (작업 디렉터리는 보통 프로젝트 루트).

엔드포인트:
    ``POST /chat`` — JSON ``{"query": "..."}`` 를 받아 답변 문자열을 반환한다.

주의:
    ``get_chat_engine()`` 는 요청마다 호출되므로, 매번 Qdrant 연결·설정 초기화가
    발생한다. 트래픽이 크면 엔진/클라이언트를 앱 수명 주기에 맞게 캐싱하는 개선을 고려한다.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.bot.llm_service import get_chat_engine

app = FastAPI()


class QueryRequest(BaseModel):
    """채팅 질의 본문. Pydantic 으로 요청 바디 검증."""

    query: str


@app.post("/chat")
async def chat(request: QueryRequest):
    """사용자 질의를 RAG 엔진에 넘기고, 응답 텍스트만 JSON 으로 돌려준다."""
    chat_engine = get_chat_engine()
    response = chat_engine.query(request.query)
    return {"answer": str(response)}

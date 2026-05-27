"""
Chainlit 기반 사내 RAG 챗봇 UI.

실행 예:
    프로젝트 루트에서 ``chainlit run src/bot/app.py`` (또는 프로젝트에 맞는 엔트리 경로).

동작:
    - 채팅 세션 시작 시 ``get_chat_engine()`` 으로 LlamaIndex 질의 엔진을 만들고
      ``cl.user_session`` 에 저장해, 이후 메시지마다 같은 엔진을 재사용한다.
    - 사용자 메시지마다 ``engine.query()`` 로 RAG 답변을 생성한다.

전제:
    Qdrant·Ollama·사전 적재된 ``corporate_knowledge`` 컬렉션은 ``llm_service`` 와 동일 조건이어야 한다.
"""
import chainlit as cl

from src.bot.llm_service import get_chat_engine

# Chainlit 은 내부적으로 FastAPI 앱을 노출한다. openapi_url 을 끄면
# Swagger/OpenAPI 문서 경로가 비활성화되어(보안·노출 최소화) 배포 시 흔히 쓰인다.
cl.server.app.openapi_url = None


@cl.on_chat_start
async def start() -> None:
    """새 채팅 스레드가 열릴 때 한 번 호출된다."""
    # 매 세션마다 엔진을 새로 만들면 Qdrant/모델 연결 비용이 들 수 있어,
    # 세션 단위로 한 번 생성 후 user_session 에 보관한다.
    engine = get_chat_engine()
    cl.user_session.set("engine", engine)
    await cl.Message(
        content="사내 지식 챗봇에 오신 것을 환영합니다! 무엇을 도와드릴까요?"
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """사용자가 메시지를 보낼 때마다 호출된다."""
    engine = cl.user_session.get("engine")
    # RAG: 내부적으로 질의 임베딩 → Qdrant 유사 검색 → 컨텍스트 + LLM 생성
    response = engine.query(message.content)

    # LlamaIndex QueryEngine 응답은 기본적으로 `response`(텍스트)와
    # `source_nodes`(어떤 문서 청크를 근거로 했는지)를 함께 가진다.
    text = str(response)
    try:
        source_nodes = getattr(response, "source_nodes", None) or []
    except Exception:
        source_nodes = []

    # 1) 메인 답변 메시지
    await cl.Message(content=text).send()

    # 2) 근거 문서 미리보기(있을 때만)
    if source_nodes:
        preview_lines = []
        for i, node in enumerate(source_nodes[:3], start=1):
            meta = node.node.metadata or {}
            filename = meta.get("filename") or meta.get("url") or meta.get("source") or "알 수 없는 출처"
            snippet = (node.node.get_content() or "")[:200].replace("\n", " ")
            preview_lines.append(f"{i}. [{filename}] {snippet}...")

        await cl.Message(
            content="다음 근거 문서를 참고해 답변을 생성했습니다:\n\n" + "\n".join(preview_lines)
        ).send()

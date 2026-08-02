"""
Chainlit 기반 사내 RAG 챗봇 UI.
"""
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chainlit as cl
from src.bot.llm_service import get_chat_engine

# 💡 [핵심] FastAPI OpenAPI 생성 시 OAuth2 스키마 충돌 오류 방지 가드
try:
    from chainlit.server import app as fastapi_app
    def custom_openapi():
        if fastapi_app.openapi_schema:
            return fastapi_app.openapi_schema
        # 기본 openapi 생성 시 에러가 발생하는 구문을 통과하도록 빈 스키마 반환
        fastapi_app.openapi_schema = {
            "openapi": "3.0.2",
            "info": {"title": "Chainlit API", "version": "1.0.0"},
            "paths": {},
        }
        return fastapi_app.openapi_schema
    fastapi_app.openapi = custom_openapi
except Exception:
    pass


@cl.on_chat_start
async def start() -> None:
    """새 채팅 스레드가 열릴 때 한 번 호출된다."""
    engine = get_chat_engine()
    cl.user_session.set("engine", engine)
    await cl.Message(
        content="사내 지식 챗봇에 오신 것을 환영합니다! 무엇을 도와드릴까요?"
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """사용자가 메시지를 보낼 때마다 호출된다."""
    engine = cl.user_session.get("engine")
    
    msg = cl.Message(content="")
    await msg.send()

    # 동기 질의(query)를 비동기로 안전하게 실행
    response = await cl.make_async(engine.query)(message.content)

    text = str(response)
    try:
        source_nodes = getattr(response, "source_nodes", None) or []
    except Exception:
        source_nodes = []

    # 근거 문서 미리보기 생성 (중복 제거 및 텍스트 깔끔 정제)
    sources_text = ""
    if source_nodes:
        preview_lines = []
        seen_pages = set()
        
        for node in source_nodes:
            meta = getattr(node.node, "metadata", {}) or {}
            filename = meta.get("filename") or "문서"
            page_label = meta.get("page_label", "?")
            
            page_key = f"{filename}_p{page_label}"
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            raw_text = node.node.get_content() or ""
            # 줄바꿈 제거 및 연속 공백 정리
            clean_snippet = " ".join(raw_text.split())[:100]
            
            preview_lines.append(f"- [{filename} (p.{page_label})]: {clean_snippet}...")
            
            if len(preview_lines) >= 3:  # 사용자 화면에는 상위 3개 출처만 깔끔히 표시
                break

        if preview_lines:
            sources_text = "\n\n**다음 근거 문서를 참고해 답변을 생성했습니다:**\n" + "\n".join(preview_lines)

    msg.content = text + sources_text
    await msg.update()
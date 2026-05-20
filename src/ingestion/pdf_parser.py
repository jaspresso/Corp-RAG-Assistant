"""
PDF 디렉터리를 스캔해 LlamaIndex ``Document`` 리스트로 변환한다.

Docling 의 ``DocumentConverter`` 로 PDF 를 구조 인식 후 Markdown 으로보내고,
파일명·출처 메타데이터를 붙여 RAG 인제션 파이프라인에 바로 넣을 수 있게 한다.

전제:
    ``pdf_dir`` 에 ``.pdf`` 확장자 파일만 대상으로 한다 (대소문자 구분: ``.pdf`` 만).
"""
import os

from docling.document_converter import DocumentConverter
from llama_index.core import Document


def parse_pdf_to_documents(pdf_dir: str) -> list[Document]:
    """
    ``pdf_dir`` 아래의 모든 PDF 를 읽어 ``Document`` 리스트로 반환한다.

    Args:
        pdf_dir: PDF 파일들이 들어 있는 디렉터리 경로 (상대/절대 가능).

    Returns:
        각 PDF 당 하나의 ``Document`` (text=마크다운 본문, metadata 에 filename/source).

    Note:
        Docling 변환은 파일 크기·페이지 수에 따라 시간이 걸릴 수 있다.
    """
    converter = DocumentConverter()
    documents: list[Document] = []

    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue
        file_path = os.path.join(pdf_dir, filename)
        print(f"📄 파싱 중: {filename}")

        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()

        doc = Document(
            text=markdown_text,
            metadata={"filename": filename, "source": "pdf"},
        )
        documents.append(doc)

    return documents

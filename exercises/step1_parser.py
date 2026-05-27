"""
Step 1 연습: PDF 디렉터리 → LlamaIndex Document 리스트.

실행 예 (프로젝트 루트):
    python -m exercises.step1_parser

미리보기 길이 변경:
    PREVIEW_CHARS=1000 python -m exercises.step1_parser
"""
import os
import sys

try:
    from docling.document_converter import DocumentConverter
    from llama_index.core import Document
except ModuleNotFoundError as exc:
    raise SystemExit(
        "필요 패키지가 현재 Python 환경에 없습니다 (10번 라인 import 실패).\n"
        "venv를 쓰세요:\n"
        "  cd 프로젝트루트\n"
        "  source venv/bin/activate\n"
        "  python -m exercises.step1_parser\n"
        "또는: ./venv/bin/python -m exercises.step1_parser\n"
        f"\n원인: {exc}"
    ) from None


def parse_pdf_to_documents(pdf_dir: str) -> list[Document]:
    """
    ``pdf_dir`` 아래 PDF 파일을 Docling으로 마크다운 변환 후 Document 로 반환한다.
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF 디렉터리가 없습니다: {pdf_dir}")

    converter = DocumentConverter()
    documents: list[Document] = []

    for filename in sorted(os.listdir(pdf_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(pdf_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            print(f"📄 파싱 중: {filename}")
            result = converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()

            documents.append(
                Document(
                    text=markdown_text,
                    metadata={"filename": filename, "source": "pdf"},
                )
            )
        except Exception as exc:
            print(f"⚠️ 건너뜀 ({filename}): {exc}", file=sys.stderr)

    return documents


def print_document_previews(documents: list[Document], max_chars: int = 500) -> None:
    """파싱 결과 미리보기: 각 Document 앞부분만 출력해 변환 품질을 확인한다."""
    for doc in documents:
        filename = doc.metadata.get("filename", "(이름 없음)")
        text = doc.text or ""
        preview = text[:max_chars]
        truncated = len(text) > max_chars
        print(f"\n--- {filename} (전체 {len(text)}자) ---")
        print(preview)
        if truncated:
            print(f"... (이하 {len(text) - max_chars}자 생략)")


if __name__ == "__main__":
    pdf_dir = os.environ.get("PDF_DIR", "./data/pdfs")
    preview_len = int(os.environ.get("PREVIEW_CHARS", "500"))

    docs = parse_pdf_to_documents(pdf_dir)
    print(f"✅ {len(docs)}개 Document 생성 완료")

    if docs:
        print_document_previews(docs, max_chars=preview_len)
    else:
        print("⚠️ 출력할 Document가 없습니다.")

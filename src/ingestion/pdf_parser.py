"""
PyMuPDF(fitz)와 pytesseract OCR을 결합하여
스캔 이미지 PDF에서도 한국어 텍스트 및 표 구조를 정밀 추출하여 Document로 변환한다.
"""
import io
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from llama_index.core import Document


def parse_pdf_to_documents(pdf_dir: str) -> list[Document]:
    """
    ``pdf_dir`` 아래의 PDF를 순회하며, 스캔 이미지 PDF일 경우 OCR 엔진을 작동시켜 
    페이지별 독립 LlamaIndex Document 리스트로 반환한다.
    """
    documents: list[Document] = []

    if not os.path.exists(pdf_dir):
        print(f"❌ [경고] 지정된 경로가 존재하지 않습니다: {pdf_dir}")
        return documents

    files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not files:
        print(f"❌ [경고] {pdf_dir} 경로 내에 PDF 파일이 없습니다.")
        return documents

    for filename in files:
        file_path = os.path.join(pdf_dir, filename)
        print(f"📄 PDF 파싱 및 OCR 분석 시작: {filename}")

        try:
            doc_obj = fitz.open(file_path)
            page_count = len(doc_obj)
            file_docs_count = 0

            for page_num in range(page_count):
                page = doc_obj[page_num]
                
                # 1. 일반 텍스트 추출 시도
                extracted_text = page.get_text("text", sort=True).strip()

                # 2. 텍스트가 없는 스캔 이미지 페이지인 경우 Tesseract OCR 작동
                if not extracted_text:
                    # 페이지를 고해상도 이미지(DPI 300)로 렌더링
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))

                    # 한국어 + 영어 OCR 수행
                    extracted_text = pytesseract.image_to_string(image, lang="kor+eng").strip()

                # 유효 텍스트 추출 성공 시 Document 생성
                if extracted_text:
                    doc = Document(
                        text=extracted_text,
                        metadata={
                            "filename": filename,
                            "page_label": str(page_num + 1),
                            "source": "pdf"
                        },
                    )
                    documents.append(doc)
                    file_docs_count += 1
                    if (page_num + 1) % 10 == 0 or page_num + 1 == page_count:
                        print(f"  └─ 진행 중: {page_num + 1}/{page_count} 페이지 완료...")

            doc_obj.close()
            print(f"🎉 {filename}: 총 {page_count}페이지 중 {file_docs_count}개 페이지 OCR 파싱 성공!")

        except Exception as e:
            print(f"❌ {filename} 파싱 중 오류 발생: {e}")

    return documents
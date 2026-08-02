"""
pdfplumber 및 Tesseract OCR 기반 고성능 PDF/표 구조 파서.
"""
import os
import glob
import pdfplumber
import pytesseract
from PIL import Image
from llama_index.core import Document


def extract_table_markdown(page) -> str:
    """페이지 내 표(Table)를 마크다운 텍스트 형태(| Col1 | Col2 |)로 정제 추출"""
    tables = page.extract_tables()
    table_texts = []
    
    for table in tables:
        if not table:
            continue
        md_lines = []
        for row_idx, row in enumerate(table):
            # None 또는 줄바꿈 정제
            clean_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
            md_line = "| " + " | ".join(clean_row) + " |"
            md_lines.append(md_line)
            
            # 표 헤더 구분선 추가
            if row_idx == 0:
                divider = "| " + " | ".join(["---"] * len(clean_row)) + " |"
                md_lines.append(divider)
                
        table_texts.append("\n".join(md_lines))
        
    return "\n\n".join(table_texts)


def parse_pdf_file(pdf_path: str) -> list[Document]:
    """단일 PDF 파일 파싱 (텍스트 + pdfplumber 표 + OCR  fallback)"""
    documents = []
    filename = os.path.basename(pdf_path)
    print(f"📄 `{filename}` 표 정밀 파싱 시작...")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            table_md = extract_table_markdown(page)

            combined_text = ""
            if page_text.strip():
                combined_text += f"[본문 텍스트]\n{page_text.strip()}\n\n"
            if table_md.strip():
                combined_text += f"[추출된 표 구조]\n{table_md.strip()}\n\n"

            # 텍스트와 표가 모두 없는 스캔 이미지 페이지인 경우 OCR 적용
            if not combined_text.strip():
                try:
                    pil_image = page.to_image(resolution=150).original
                    ocr_text = pytesseract.image_to_string(pil_image, lang="kor+eng")
                    combined_text = f"[OCR 스캔 텍스트]\n{ocr_text.strip()}"
                except Exception as e:
                    print(f"⚠️ {page_num}페이지 OCR 추출 중 경고: {e}")

            if combined_text.strip():
                doc = Document(
                    text=combined_text.strip(),
                    metadata={
                        "filename": filename,
                        "page_label": str(page_num),
                        "total_pages": total_pages,
                        "source": pdf_path,
                    },
                )
                documents.append(doc)

            if page_num % 50 == 0 or page_num == total_pages:
                print(f"  └─ 진행 중: {page_num}/{total_pages} 페이지 파싱 완료...")

    return documents


def parse_pdf_to_documents(pdf_dir: str) -> list[Document]:
    """지정된 디렉터리의 모든 PDF 파싱"""
    all_documents = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"⚠️ `{pdf_dir}` 경로에 PDF 파일이 없습니다.")
        return all_documents

    for pdf_path in pdf_files:
        docs = parse_pdf_file(pdf_path)
        all_documents.extend(docs)
        print(f"🎉 `{os.path.basename(pdf_path)}`: 총 {len(docs)}개 페이지 추출 완료!")

    return all_documents
#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

# 프로젝트 루트 디렉터리로 이동
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo "🚀 RAG Corporate Chatbot 시스템 구동을 시작합니다."
echo "=================================================="

# 1. 가상환경 확인
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
VENV_CHAINLIT="$PROJECT_DIR/venv/bin/chainlit"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 가상환경(venv)을 찾을 수 없습니다. 파이썬 가상환경을 먼저 생성해 주세요."
    exit 1
fi

# 2. 필수 디렉터리 생성
mkdir -p ./data/pdfs

# 3. Qdrant 및 Ollama 서비스 동작 여부 점검 및 자동 실행
echo "🔍 필수 백엔드 서비스(Qdrant, Ollama) 상태 점검 중..."

if ! curl -s http://localhost:6333/info > /dev/null; then
    echo "⚠️ Qdrant가 꺼져 있습니다. Docker 컨테이너 재시작을 시도합니다..."
    docker start qdrant 2>/dev/null || docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z --name qdrant qdrant/qdrant
    sleep 3
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama 서비스가 실행되어 있지 않습니다. (http://localhost:11434)"
    echo "💡 터미널에서 'ollama serve'를 실행해 주세요."
    exit 1
fi

echo "✅ 백엔드 서비스 정상 작동 확인 완료!"

# 4. PDF OCR 및 Qdrant 데이터 적재 파이프라인 실행
echo "=================================================="
echo "📥 PDF OCR 수집 및 Qdrant (v11) 데이터 적재 파이프라인 실행"
echo "=================================================="
$VENV_PYTHON -m src.ingestion.pipeline

# 5. Chainlit 챗봇 서버 기동 (-h 옵션으로 xdg-open 브라우저 자동 오픈 에러 방지)
echo "=================================================="
echo "🤖 Chainlit RAG 챗봇 웹 서버를 시작합니다..."
echo "=================================================="
$VENV_CHAINLIT run src/bot/app.py -h

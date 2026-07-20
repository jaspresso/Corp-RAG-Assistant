#!/usr/bin/env bash
#
# Corp RAG Assistant — Qdrant(Docker) + (필요 시) PDF 적재 + Chainlit UI 기동
#
# 사용 (프로젝트 루트에서):
#   bash scripts/start.sh
#   bash scripts/start.sh --force-ingest   # PDF 재적재 후 실행
#   bash scripts/start.sh --skip-ingest    # 적재 생략 (컬렉션 없으면 채팅 실패 가능)
#
# 브라우저: http://localhost:8000
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE_INGEST=0
SKIP_INGEST=0
CHAINLIT_PORT="${CHAINLIT_PORT:-8000}"

for arg in "$@"; do
  case "$arg" in
    --force-ingest) FORCE_INGEST=1 ;;
    --skip-ingest) SKIP_INGEST=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $arg (--help 참고)"
      exit 1
      ;;
  esac
done

log() { echo "[start] $*"; }
warn() { echo "[start][WARN] $*" >&2; }
die() { echo "[start][ERROR] $*" >&2; exit 1; }

# --- Docker / Qdrant ---
if ! command -v docker >/dev/null 2>&1; then
  die "docker 가 설치되어 있지 않습니다. Docker Desktop(WSL2) 또는 docker.io 를 설치하세요."
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  die "docker compose 를 사용할 수 없습니다."
fi

mkdir -p storage/qdrant_data data/pdfs

qdrant_ready() {
  curl -sf "http://localhost:6333/readyz" >/dev/null 2>&1 \
    || curl -sf "http://localhost:6333/collections" >/dev/null 2>&1
}

port_in_use() {
  ss -tln 2>/dev/null | grep -q ':6333 ' \
    || netstat -tln 2>/dev/null | grep -q ':6333 '
}

ensure_qdrant() {
  if qdrant_ready; then
    log "Qdrant 가 이미 실행 중입니다 (http://localhost:6333) — Docker 기동을 건너뜁니다."
    return 0
  fi

  if port_in_use; then
    die "6333 포트는 사용 중이지만 Qdrant API 가 응답하지 않습니다.
  점유 프로세스 확인: ss -tlnp | grep 6333
  또는: docker ps -a | grep qdrant
  기존 Qdrant 를 중지한 뒤 다시 실행하거나, 다른 포트로 띄우도록 docker-compose.yml 을 수정하세요."
  fi

  log "Qdrant Docker 컨테이너 기동..."
  if ! "${COMPOSE[@]}" up -d qdrant; then
    if qdrant_ready; then
      warn "Docker up 은 실패했지만 Qdrant API 가 응답합니다. 기존 인스턴스를 사용합니다."
      return 0
    fi
    die "Qdrant 컨테이너 기동 실패. docker logs corp-rag-qdrant 확인"
  fi

  log "Qdrant 준비 대기 (http://localhost:6333)..."
  for i in $(seq 1 45); do
    if qdrant_ready; then
      log "Qdrant 준비 완료."
      return 0
    fi
    if [[ "$i" -eq 45 ]]; then
      die "Qdrant 가 45초 내에 응답하지 않습니다. docker logs corp-rag-qdrant 확인"
    fi
    sleep 1
  done
}

ensure_qdrant

# --- Python venv ---
VENV_PY="$ROOT/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  die "venv 가 없습니다. 먼저: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
fi

# --- Ollama (로컬, Docker 아님) ---
if curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  log "Ollama 연결 확인됨 (http://localhost:11434)."
else
  warn "Ollama(http://localhost:11434)에 연결되지 않습니다."
  warn "  다른 터미널에서: ollama serve"
  warn "  모델: ollama pull llama3"
fi

# --- 컬렉션 존재 여부 ---
collection_exists() {
  curl -sf "http://localhost:6333/collections/corporate_knowledge/exists" \
    | grep -q '"exists":true'
}

run_pdf_ingest() {
  log "PDF → Qdrant 적재 중 (data/pdfs)..."
  "$VENV_PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from src.ingestion.pdf_parser import parse_pdf_to_documents
from src.ingestion.pipeline import run_full_pipeline

pdf_dir = "./data/pdfs"
docs = parse_pdf_to_documents(pdf_dir)
if not docs:
    raise SystemExit(
        f"적재할 PDF 가 없습니다: {pdf_dir}\n"
        "PDF 파일을 넣은 뒤 다시 실행하거나 --force-ingest 로 재시도하세요."
    )
run_full_pipeline(docs)
print(f"적재 완료: {len(docs)}개 문서")
PY
}

if [[ "$SKIP_INGEST" -eq 1 ]]; then
  log "적재 생략 (--skip-ingest)."
elif [[ "$FORCE_INGEST" -eq 1 ]]; then
  run_pdf_ingest
elif collection_exists; then
  log "Qdrant 컬렉션 'corporate_knowledge' 이미 존재 — 적재 생략."
  log "  PDF 를 다시 넣었다면: bash scripts/start.sh --force-ingest"
else
  log "컬렉션이 없어 PDF 적재를 실행합니다..."
  run_pdf_ingest
fi

# --- Chainlit ---
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

log "Chainlit UI 기동 → http://localhost:${CHAINLIT_PORT}"
log "종료: Ctrl+C"

if [[ -x "$ROOT/venv/bin/chainlit" ]]; then
  exec "$ROOT/venv/bin/chainlit" run src/bot/app.py -w --port "$CHAINLIT_PORT"
else
  exec "$VENV_PY" -m chainlit run src/bot/app.py -w --port "$CHAINLIT_PORT"
fi

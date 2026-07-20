#!/usr/bin/env bash
# Qdrant Docker 컨테이너만 중지 (Chainlit 은 별도 Ctrl+C)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if docker compose version >/dev/null 2>&1; then
  docker compose stop qdrant
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose stop qdrant
else
  echo "docker compose 를 찾을 수 없습니다." >&2
  exit 1
fi

echo "[stop] Qdrant 컨테이너 중지됨."

#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Datewise RAG API Startup ==="

# Start Qdrant
echo "Starting Qdrant..."
docker compose up -d

echo "Waiting for Qdrant to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Qdrant did not become ready in time."
        exit 1
    fi
    sleep 2
done

COLLECTION="${DATEWISE_COLLECTION:-datewise_transcripts}"
echo "Ensuring Qdrant collection exists: ${COLLECTION}"
.venv/bin/python -c "
import os; import sys; sys.path.insert(0, 'src')
from dating_rag.store.qdrant import QdrantStore
collection = os.environ.get('DATEWISE_COLLECTION', 'datewise_transcripts')
store = QdrantStore(url='http://localhost:6333')
if not store.collection_exists(collection):
    store.create_collection(collection, dense_vector_size=1024)
    print(f'Created {collection} collection')
else:
    print(f'{collection} collection already exists')
"

echo "Starting Datewise RAG API..."
exec .venv/bin/python -m uvicorn dating_rag.api.app:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --reload

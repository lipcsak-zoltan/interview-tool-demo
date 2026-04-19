#!/usr/bin/env python3
"""Rebuild the synthetic Chroma database from the validated JSONL dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_dataset import DEFAULT_DATASET_PATH, load_rows, validate_rows

DEFAULT_DB_DIR = ROOT / "db" / "chroma_demo"
COLLECTION_NAME = "demo_interviews"
EMBEDDING_MODEL = "text-embedding-3-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to rebuild embeddings.")

    rows = load_rows(args.dataset)
    errors = validate_rows(rows)
    if errors:
        raise SystemExit("Dataset validation failed; run scripts/validate_dataset.py for details.")

    import chromadb
    from demo_embeddings import OpenAIEmbeddingFunction

    if args.db_dir.exists():
        shutil.rmtree(args.db_dir)
    args.db_dir.mkdir(parents=True, exist_ok=True)

    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    settings = chromadb.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(args.db_dir), settings=settings)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.add(
            ids=[row["id"] for row in batch],
            documents=[row["text"] for row in batch],
            metadatas=[
                {
                    "site": row["site"],
                    "question_no": row["question_no"],
                    "interviewee_no": row["interviewee_no"],
                    "collar": row["collar"],
                    "role": row["role"],
                }
                for row in batch
            ],
        )

    print(f"Rebuilt {COLLECTION_NAME} with {collection.count()} chunks at {args.db_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

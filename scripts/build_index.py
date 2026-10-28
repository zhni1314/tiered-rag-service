"""Build or rebuild the vector index from a folder of PDFs, without the HTTP layer.

    python scripts/build_index.py --dir data/pdfs --level public
    python scripts/build_index.py --dir data/secret_docs --level secret --reset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from tiered_rag.bootstrap import configure_huggingface  # noqa: E402
from tiered_rag.config import Settings  # noqa: E402
from tiered_rag.indexer import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="folder containing PDF files")
    parser.add_argument("--level", default="public", help="permission tier to tag chunks with")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.level not in settings.levels:
        parser.error(f"--level must be one of {settings.levels}")
    configure_huggingface(settings)

    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    if args.reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        print(f"🗑  removed {settings.chroma_dir}")

    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=HuggingFaceEmbeddings(model_name=settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
    )

    pdfs = sorted(args.dir.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found under {args.dir}")
        return 1

    total = 0
    for pdf in pdfs:
        docs = ingest(pdf, args.level, settings)
        if not docs:
            print(f"⏭  {pdf.name}: no extractable text (scanned PDF? needs OCR)")
            continue
        vectorstore.add_documents(docs)
        total += len(docs)
        print(f"✅ {pdf.name}: {len(docs)} chunks")

    print(f"\n{len(pdfs)} files, {total} chunks indexed at level {args.level!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

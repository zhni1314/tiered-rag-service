"""Build or rebuild the vector index from a folder of PDFs, without the HTTP layer.

    python scripts/build_index.py --dir data/pdfs --level public
    python scripts/build_index.py --dir data/secret_docs --level secret --reset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Windows consoles default to GBK; an unencodable file name must not kill a
# half-finished index build.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from tiered_rag.bootstrap import configure_huggingface  # noqa: E402
from tiered_rag.config import Settings  # noqa: E402
from tiered_rag.indexer import ingest  # noqa: E402
from tiered_rag.parents import ParentStore  # noqa: E402


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

    if args.reset:
        if settings.chroma_dir.exists():
            shutil.rmtree(settings.chroma_dir)
            print(f"reset: removed {settings.chroma_dir}")
        settings.parent_store.unlink(missing_ok=True)

    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=HuggingFaceEmbeddings(model_name=settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
    )

    pdfs = sorted(args.dir.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found under {args.dir}")
        return 1

    parents = ParentStore(settings.parent_store)
    parents.load()

    total = 0
    for pdf in pdfs:
        result = ingest(pdf, args.level, settings)
        if not result.children:
            print(f"skip: {pdf.name}: no extractable text (scanned PDF? needs OCR)")
            continue
        # Re-indexing one file replaces its old chunks instead of duplicating them.
        stale_ids = vectorstore.get(where={"source": pdf.name}).get("ids", [])
        vectorstore.add_documents(result.children)
        if stale_ids:
            vectorstore.delete(ids=stale_ids)
        parents.update(pdf.name, result.parents)
        total += len(result.children)
        print(
            f"indexed: {pdf.name}: {len(result.children)} children / {len(result.parents)} parents"
        )

    print(f"\n{len(pdfs)} files, {total} chunks indexed at level {args.level!r}")
    print(f"parents stored in {settings.parent_store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

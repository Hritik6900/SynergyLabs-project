"""Ingestion: load PDF/HTML/MD from a folder, chunk, and upsert into the store.

Key properties:
  * Supports .pdf, .html/.htm, .md/.markdown, and plain .txt.
  * Chunk size + overlap are configurable (defaults in config).
  * Deterministic, content-addressed chunk IDs make re-ingestion idempotent:
    an id is sha256(source_path + "::" + chunk_text). Before inserting we ask the
    store which of the candidate ids already exist and skip those, so running
    ingestion twice never creates duplicate vectors.
  * Every chunk carries metadata: source filename, page/section, chunk_index.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .chunker import Chunk, chunk_text
from .config import settings
from .embed_store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}


@dataclass
class LoadedDocument:
    """One source document, possibly split into pages/sections before chunking."""

    source_path: str
    filename: str
    # List of (section_label, text). PDFs produce one entry per page
    # ("page 1", ...); HTML/MD/TXT produce a single ("full", text) entry.
    sections: list[tuple[str, str]]


@dataclass
class IngestStats:
    files_processed: int = 0
    files_skipped_unsupported: int = 0
    chunks_total: int = 0
    chunks_inserted: int = 0
    chunks_skipped_existing: int = 0
    per_file: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_skipped_unsupported": self.files_skipped_unsupported,
            "chunks_total": self.chunks_total,
            "chunks_inserted": self.chunks_inserted,
            "chunks_skipped_existing": self.chunks_skipped_existing,
            "per_file": self.per_file,
        }


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def _load_pdf(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(str(path))
    sections: list[tuple[str, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append((f"page {page_num}", text))
    return sections


def _load_html(path: Path) -> list[tuple[str, str]]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Prefer the document <title> as the section label when present.
    title = soup.title.get_text(strip=True) if soup.title else "full"
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines produced by block tags.
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [(title or "full", text)] if text.strip() else []


def _load_text(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [("full", text)] if text.strip() else []


def load_document(path: Path) -> LoadedDocument | None:
    """Load a single file into a :class:`LoadedDocument`, or None if unsupported."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        sections = _load_pdf(path)
    elif ext in {".html", ".htm"}:
        sections = _load_html(path)
    elif ext in {".md", ".markdown", ".txt"}:
        sections = _load_text(path)
    else:
        return None
    return LoadedDocument(
        source_path=str(path),
        filename=path.name,
        sections=sections,
    )


# --------------------------------------------------------------------------- #
# Deterministic chunk id                                                       #
# --------------------------------------------------------------------------- #
def compute_chunk_id(source_path: str, chunk_text_value: str) -> str:
    """Content-addressed id: sha256(source_path + "::" + chunk_text).

    Deterministic in both inputs, so identical content in the same file always
    maps to the same id — the basis for idempotent re-ingestion.
    """
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"::")
    h.update(chunk_text_value.encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Ingest                                                                       #
# --------------------------------------------------------------------------- #
def _chunk_document(doc: LoadedDocument) -> list[tuple[str, Chunk, str]]:
    """Return (chunk_id, Chunk, section_label) tuples for one document.

    chunk_index is assigned globally across the document (0..N-1) so it is unique
    and stable per source file even when the file has multiple pages/sections.
    """
    out: list[tuple[str, Chunk, str]] = []
    global_index = 0
    for section_label, section_text in doc.sections:
        pieces = chunk_text(
            section_text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        for piece in pieces:
            # Reindex to a document-global counter.
            piece = Chunk(
                text=piece.text,
                chunk_index=global_index,
                token_count=piece.token_count,
            )
            cid = compute_chunk_id(doc.source_path, piece.text)
            out.append((cid, piece, section_label))
            global_index += 1
    return out


def ingest_folder(folder: str, store: VectorStore | None = None) -> IngestStats:
    """Ingest every supported file under ``folder`` into the vector store.

    Idempotent: chunks whose deterministic id already exists in the store are
    skipped rather than re-inserted, so no duplicate vectors are created.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Ingestion folder does not exist: {folder}")

    store = store or VectorStore()
    stats = IngestStats()

    paths = sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            stats.files_skipped_unsupported += 1
            continue

        doc = load_document(path)
        if doc is None or not doc.sections:
            stats.files_skipped_unsupported += 1
            continue

        chunk_tuples = _chunk_document(doc)
        if not chunk_tuples:
            stats.files_processed += 1
            stats.per_file.append(
                {"file": doc.filename, "chunks_total": 0, "inserted": 0, "skipped": 0}
            )
            continue

        ids = [cid for cid, _, _ in chunk_tuples]

        # --- Idempotency check: which ids are already stored? ---
        existing = store.existing_ids(ids)

        new_ids: list[str] = []
        new_texts: list[str] = []
        new_metas: list[dict] = []
        for cid, piece, section_label in chunk_tuples:
            if cid in existing:
                continue
            new_ids.append(cid)
            new_texts.append(piece.text)
            new_metas.append(
                {
                    "source": doc.filename,
                    "source_path": doc.source_path,
                    "section": section_label,  # page/section within the source
                    "chunk_index": piece.chunk_index,
                    "token_count": piece.token_count,
                }
            )

        if new_ids:
            store.add(ids=new_ids, texts=new_texts, metadatas=new_metas)

        inserted = len(new_ids)
        skipped = len(chunk_tuples) - inserted
        stats.files_processed += 1
        stats.chunks_total += len(chunk_tuples)
        stats.chunks_inserted += inserted
        stats.chunks_skipped_existing += skipped
        stats.per_file.append(
            {
                "file": doc.filename,
                "chunks_total": len(chunk_tuples),
                "inserted": inserted,
                "skipped": skipped,
            }
        )

    return stats


if __name__ == "__main__":  # pragma: no cover - manual CLI use
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest a folder into ChromaDB.")
    parser.add_argument("folder", help="Path to the corpus folder")
    args = parser.parse_args()

    result = ingest_folder(args.folder)
    print(json.dumps(result.as_dict(), indent=2))
    print(f"\nCollection now holds {VectorStore().count()} chunks "
          f"(persist dir: {os.path.abspath(settings.chroma_persist_dir)})")

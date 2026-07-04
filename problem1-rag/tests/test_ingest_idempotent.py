"""Ingestion: deterministic ids, idempotent re-ingest, metadata."""

from __future__ import annotations

from src.ingest import compute_chunk_id, ingest_folder


def test_chunk_id_is_deterministic_and_content_sensitive():
    a = compute_chunk_id("docs/x.md", "hello world")
    b = compute_chunk_id("docs/x.md", "hello world")
    c = compute_chunk_id("docs/x.md", "hello  world")   # different text
    d = compute_chunk_id("docs/y.md", "hello world")    # different path
    assert a == b            # same inputs -> same id
    assert a != c and a != d # any input change -> different id


def _write_corpus(dirpath):
    (dirpath / "a.md").write_text("# Doc A\n\nThe quick brown fox jumps over the lazy dog.")
    (dirpath / "b.txt").write_text("Completely different content about vector databases.")


def test_reingest_is_idempotent(tmp_path, store):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus(corpus)

    first = ingest_folder(str(corpus), store=store)
    assert first.chunks_inserted > 0
    count_after_first = store.count()
    assert count_after_first == first.chunks_inserted

    second = ingest_folder(str(corpus), store=store)
    assert second.chunks_inserted == 0                      # nothing new
    assert second.chunks_skipped_existing == first.chunks_inserted
    assert store.count() == count_after_first               # count unchanged


def test_metadata_is_recorded(tmp_path, store):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus(corpus)
    ingest_folder(str(corpus), store=store)

    rows = store.all_chunks()
    sources = {r["source"] for r in rows}
    assert sources == {"a.md", "b.txt"}
    for r in rows:
        assert r["chunk_index"] is not None
        assert r["section"] is not None

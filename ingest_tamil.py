"""
ingest_tamil.py — Tamil-only ingestion for Habeas Corpus
=========================================================
Processes only sc_tamil/ PDFs into output/, Neo4j, and FAISS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz

from extractor.clean_text import clean_document
from extractor.entity_extractor import extract_legal_graph
from graph.graph_builder import build_graph
from graph.neo4j_loader import load_graph
from semantic_retrieval.pipeline import build_index

_PROJECT_ROOT = Path(__file__).parent
_SC_TAMIL     = _PROJECT_ROOT / "sc_tamil"
_OUTPUT_DIR   = _PROJECT_ROOT / "output"


def _process_single_pdf(pdf_path: Path, output_dir: Path, dry_run: bool) -> str:
    txt_path = output_dir / f"{pdf_path.stem}.txt"

    if txt_path.exists() and txt_path.stat().st_size > 0:
        return "cached"

    try:
        doc = fitz.open(pdf_path)
        raw_text = "\n".join([page.get_text() for page in doc])
        doc.close()

        if not raw_text.strip():
            return "empty"

        cleaned_text = clean_document(raw_text)
        txt_path.write_text(cleaned_text, encoding="utf-8")

        legal_graph = extract_legal_graph(cleaned_text)
        legal_graph["nodes"]["case"]["language"] = "Tamil"

        loader_graph = build_graph(legal_graph)
        load_graph(loader_graph, dry_run=dry_run)

        return "ok"

    except Exception as exc:
        return f"error: {exc}"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    dry_run = "--live" not in sys.argv
    workers = 8

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(_SC_TAMIL.glob("*.pdf"))
    total = len(pdf_files)

    print("=" * 60)
    print("  HABEAS CORPUS — TAMIL CORPUS INGESTION")
    print("=" * 60)
    print(f"Mode    : {'DRY RUN' if dry_run else 'LIVE — loading to Neo4j'}")
    print(f"PDFs    : {total} Tamil judgments in sc_tamil/")
    print(f"Workers : {workers} parallel threads")
    print()

    success = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_pdf, p, _OUTPUT_DIR, dry_run): p for p in pdf_files}
        for done, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            result = future.result()
            if result in ("ok", "cached"):
                success += 1
            elif result.startswith("error"):
                print(f"  [ERROR] {pdf.name}: {result}")

            if done % 20 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / max(elapsed, 0.001)
                print(f"  [{done:4d}/{total}] {rate:.1f} PDFs/sec  elapsed: {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"\n[ingest_tamil] Complete: {success}/{total} Tamil PDFs in {elapsed:.1f}s")

    print("\n[ingest_tamil] Building FAISS vector index with GPU embeddings...")
    build_index()

    print("\n" + "=" * 60)
    print(f"[ingest_tamil] DONE in {(time.time() - start) / 60:.2f} minutes!")
    print("=" * 60)

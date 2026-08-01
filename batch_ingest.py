"""
batch_ingest.py — Habeas Corpus Multi-Corpus Batch Ingestion Engine
===================================================================
Discovers and ingests all Supreme Court judgment PDFs from sc_English/ and sc_tamil/
into cleaned .txt files in output/, extracts legal graph entities, and rebuilds
the FAISS BGE-M3 vector database.

Usage:
    uv run python batch_ingest.py
    uv run python batch_ingest.py --live      (to also load into live Neo4j)
    uv run python batch_ingest.py --limit 50  (to process first 50 files for testing)
"""

from __future__ import annotations

import sys
import time
import os
import json
from pathlib import Path
import fitz

from extractor.clean_text import clean_document
from extractor.entity_extractor import extract_legal_graph
from graph.graph_builder import build_graph
from graph.neo4j_loader import load_graph
from semantic_retrieval.pipeline import build_index

_PROJECT_ROOT = Path(__file__).parent
_SC_ENGLISH   = _PROJECT_ROOT / "sc_English"
_SC_TAMIL     = _PROJECT_ROOT / "sc_tamil"
_OUTPUT_DIR   = _PROJECT_ROOT / "output"


def process_pdf_folder(
    folder_path: Path,
    output_dir: Path = _OUTPUT_DIR,
    dry_run: bool = True,
    limit: int | None = None,
) -> tuple[int, int]:
    """
    Extract text and entities from all PDFs in a given folder.
    Returns (processed_count, total_count).
    """
    if not folder_path.exists():
        print(f"[batch_ingest] Skipping missing directory: {folder_path}")
        return (0, 0)

    pdf_files = sorted(folder_path.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"\n[batch_ingest] Found {len(pdf_files)} PDF(s) in '{folder_path.name}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    start_time = time.time()

    for idx, pdf_path in enumerate(pdf_files, 1):
        txt_filename = f"{pdf_path.stem}.txt"
        txt_path = output_dir / txt_filename

        # Skip if already extracted
        if txt_path.exists() and txt_path.stat().st_size > 0:
            success_count += 1
            if idx % 100 == 0 or idx == len(pdf_files):
                print(f"  [{idx:5d}/{len(pdf_files)}] (cached) {pdf_path.name}")
            continue

        try:
            doc = fitz.open(pdf_path)
            raw_text = "\n".join([page.get_text() for page in doc])
            doc.close()

            if not raw_text.strip():
                continue

            cleaned_text = clean_document(raw_text)
            txt_path.write_text(cleaned_text, encoding="utf-8")

            # Graph entity extraction
            legal_graph = extract_legal_graph(cleaned_text)
            if "sc_tamil" in str(folder_path).lower() or "_TAM" in pdf_path.name:
                legal_graph["nodes"]["case"]["language"] = "Tamil"

            loader_graph = build_graph(legal_graph)
            load_graph(loader_graph, dry_run=dry_run)

            success_count += 1
            if idx % 50 == 0 or idx == len(pdf_files):
                elapsed = time.time() - start_time
                rate = idx / max(elapsed, 0.001)
                print(f"  [{idx:5d}/{len(pdf_files)}] Extracted {pdf_path.name} ({rate:.1f} PDFs/sec)")

        except Exception as exc:
            print(f"  [ERROR] Failed processing {pdf_path.name}: {exc}")

    elapsed = time.time() - start_time
    print(f"[batch_ingest] Folder '{folder_path.name}' complete: {success_count}/{len(pdf_files)} PDFs ready in {elapsed:.2f}s")
    return success_count, len(pdf_files)


def run_full_ingestion(dry_run: bool = True, limit: int | None = None) -> None:
    """
    Main entry point for multi-corpus batch ingestion.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    print("=" * 65)
    print("      HABEAS CORPUS — MULTI-CORPUS BATCH INGESTION ENGINE      ")
    print("=" * 65)
    print(f"Mode : {'DRY RUN (Graph not written)' if dry_run else 'LIVE (Graph loaded to Neo4j)'}")
    if limit:
        print(f"Limit: First {limit} files per folder")

    start_total = time.time()

    # 1. Process sc_English and sc_tamil
    eng_ok, eng_total = process_pdf_folder(_SC_ENGLISH, dry_run=dry_run, limit=limit)
    tam_ok, tam_total = process_pdf_folder(_SC_TAMIL, dry_run=dry_run, limit=limit)

    total_ready = eng_ok + tam_ok
    total_found = eng_total + tam_total

    print("\n" + "-" * 65)
    print(f"[batch_ingest] Text & Entity Extraction Complete: {total_ready} / {total_found} total PDFs ready.")
    print("-" * 65)

    # 2. Rebuild BGE-M3 FAISS Semantic Vector Index across output/*.txt
    print("\n[batch_ingest] Building / Updating FAISS Vector Database with BGE-M3 embeddings...")
    build_index()

    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print(f"[batch_ingest] INGESTION COMPLETE in {total_time / 60:.2f} minutes!")
    print("=" * 65)


if __name__ == "__main__":
    args = sys.argv[1:]
    dry_run = "--live" not in args
    limit = None

    for i, arg in enumerate(args):
        if arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])

    run_full_ingestion(dry_run=dry_run, limit=limit)

"""
main.py — Habeas Corpus Legal Search Engine
============================================

Full ETL pipeline:

    ocr/pipeline.py         ← auto-routes: selectable text OR full OCR
          │
          ▼
    extractor/entity_extractor.py
          │
          ▼
    graph/graph_builder.py
          │
          ▼
    graph/neo4j_loader.py

Run:
    uv run python main.py                          # dry run, all output/*.txt files
    uv run python main.py --live                   # live write to Neo4j, all files
    uv run python main.py path/to/file.txt         # single pre-extracted .txt file
    uv run python main.py path/to/file.pdf         # PDF (auto-detects text vs scanned)
    uv run python main.py path/to/scan.jpg         # image file (OCR)
    uv run python main.py path/to/file.pdf --live  # single file + live Neo4j write
    uv run python main.py --engine easy            # force EasyOCR engine
    uv run python main.py --engine mock            # mock engine (pipeline test, no ML)
"""

import json
import sys
from pathlib import Path

from extractor.clean_text       import clean_document
from extractor.entity_extractor import extract_legal_graph
from graph.graph_builder        import build_graph
from graph.neo4j_loader         import load_graph

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).parent
OUTPUT_DIR     = PROJECT_ROOT / "output"
SC_ENGLISH_DIR = PROJECT_ROOT / "sc_English"

# ---------------------------------------------------------------------------
# Supported input types
# ---------------------------------------------------------------------------

_OCR_EXTENSIONS   = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
_TEXT_EXTENSIONS  = {".txt"}


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _extract_text(source_path: Path, engine: str = "paddle") -> str:
    """
    Extract clean text from any supported source.

    - .txt  — read directly (already extracted)
    - .pdf / image — route through ocr/pipeline.py (auto-detects selectable vs scanned)
    """
    if source_path.suffix.lower() in _TEXT_EXTENSIONS:
        return source_path.read_text(encoding="utf-8")

    if source_path.suffix.lower() in _OCR_EXTENSIONS:
        from ocr.pipeline import OCRConfig, extract_text as ocr_extract
        config = OCRConfig(engine_preference=engine)
        result = ocr_extract(source_path, config=config)
        return result.text

    raise ValueError(f"Unsupported file type: {source_path.suffix}")


def process_file(
    source_path: Path,
    dry_run: bool = True,
    engine: str = "paddle",
) -> dict:
    """Run the full pipeline for a single file (txt, PDF, or image)."""
    print(f"\n{'=' * 60}")
    print(f"[main] Processing: {source_path.name}")
    print(f"{'=' * 60}")

    # Stage 1 — extract / OCR
    raw_text = _extract_text(source_path, engine=engine)

    # Stage 2 — clean
    cleaned = clean_document(raw_text)
    print(f"[clean_text]  {len(raw_text):,} -> {len(cleaned):,} characters")

    # Stage 3 — extract graph entities
    legal_graph = extract_legal_graph(cleaned)
    case_name   = legal_graph["nodes"]["case"]["name"]
    print(f"[extractor]   case: {case_name or '<not found>'}")
    print(f"              judges:   {len(legal_graph['nodes']['judges'])}")
    print(f"              acts:     {len(legal_graph['nodes']['acts'])}")
    print(f"              sections: {len(legal_graph['nodes']['sections'])}")
    print(f"              parties:  {len(legal_graph['nodes']['parties'])}")
    print(f"              citations:{len(legal_graph['nodes']['citations'])}")

    # Stage 4 — validate graph
    loader_graph = build_graph(legal_graph)

    # Stage 5 — load into Neo4j (or dry run)
    load_graph(loader_graph, dry_run=dry_run)

    return loader_graph


def process_all_txt_files(dry_run: bool = True) -> None:
    """Process every .txt file in the output/ directory."""
    txt_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[main] No .txt files found in {OUTPUT_DIR}")
        return

    print(f"[main] Found {len(txt_files)} .txt file(s) in {OUTPUT_DIR}")
    graphs = []
    for txt_path in txt_files:
        graph = process_file(txt_path, dry_run=dry_run)
        graphs.append(graph)

    print(f"\n[main] Done — processed {len(graphs)} file(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args    = sys.argv[1:]
    dry_run = "--live" not in args
    engine  = "paddle"

    # Parse --engine <name>
    for i, arg in enumerate(args):
        if arg == "--engine" and i + 1 < len(args):
            engine = args[i + 1]

    # Collect positional file arguments (not flags)
    file_args = [
        a for i, a in enumerate(args)
        if not a.startswith("--") and (i == 0 or args[i - 1] != "--engine")
    ]

    if file_args:
        # Single file mode
        source_path = Path(file_args[0])
        if not source_path.exists():
            print(f"[ERROR] File not found: {source_path}")
            sys.exit(1)
        result = process_file(source_path, dry_run=dry_run, engine=engine)
        print("\n[main] Graph JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Batch mode — all .txt files in output/
        process_all_txt_files(dry_run=dry_run)

"""
main.py — Habeas Corpus Legal Search Engine
============================================

Full ETL pipeline:

    pdf_reader.py  →  clean_text.py  →  entity_extractor.py
                                               ↓
                                        graph_builder.py
                                               ↓
                                        neo4j_loader.py

Run:
    uv run python main.py                          # dry run (no DB write)
    uv run python main.py --live                   # live write to Neo4j
    uv run python main.py path/to/file.txt         # single .txt file
    uv run python main.py path/to/file.txt --live  # single .txt + live write
"""

import json
import sys
from pathlib import Path

from extractor.clean_text     import clean_document
from extractor.entity_extractor import extract_legal_graph
from graph.graph_builder       import build_graph
from graph.neo4j_loader        import load_graph

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parent
OUTPUT_DIR    = PROJECT_ROOT / "output"
SC_ENGLISH_DIR = PROJECT_ROOT / "sc_English"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def process_txt_file(txt_path: Path, dry_run: bool = True) -> dict:
    """Run the full pipeline for a single .txt file."""
    print(f"\n{'=' * 60}")
    print(f"[main] Processing: {txt_path.name}")
    print(f"{'=' * 60}")

    raw_text = txt_path.read_text(encoding="utf-8")

    # Stage 1 — clean
    cleaned = clean_document(raw_text)
    print(f"[clean_text]  {len(raw_text):,} -> {len(cleaned):,} characters")

    # Stage 2 — extract
    legal_graph = extract_legal_graph(cleaned)
    case_name   = legal_graph["nodes"]["case"]["name"]
    print(f"[extractor]   case: {case_name or '<not found>'}")
    print(f"              judges:   {len(legal_graph['nodes']['judges'])}")
    print(f"              acts:     {len(legal_graph['nodes']['acts'])}")
    print(f"              sections: {len(legal_graph['nodes']['sections'])}")
    print(f"              parties:  {len(legal_graph['nodes']['parties'])}")
    print(f"              citations:{len(legal_graph['nodes']['citations'])}")

    # Stage 3 — build / validate
    loader_graph = build_graph(legal_graph)

    # Stage 4 — load into Neo4j (or dry run)
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
        graph = process_txt_file(txt_path, dry_run=dry_run)
        graphs.append(graph)

    print(f"\n[main] Done — processed {len(graphs)} file(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args     = sys.argv[1:]
    dry_run  = "--live" not in args
    txt_args = [a for a in args if not a.startswith("--")]

    if txt_args:
        # Single file mode
        txt_path = Path(txt_args[0])
        if not txt_path.exists():
            print(f"[ERROR] File not found: {txt_path}")
            sys.exit(1)
        result = process_txt_file(txt_path, dry_run=dry_run)
        print("\n[main] Graph JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Batch mode — all .txt files in output/
        process_all_txt_files(dry_run=dry_run)

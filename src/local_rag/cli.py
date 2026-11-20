import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .embedding import E5Embedder
from .index import build_index, index_status, search_index


def main() -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Index and search local documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Rebuild the document index")
    index.add_argument("--input", type=Path, required=True, help="Directory of UTF-8 Markdown and text files")
    index.add_argument("--index", type=Path, required=True, help="SQLite output path")

    status = commands.add_parser("status", help="Show index size and available scopes as JSON")
    status.add_argument("--index", type=Path, required=True, help="SQLite index path")

    search = commands.add_parser("search", help="Return the most relevant source passages")
    search.add_argument("--index", type=Path, required=True, help="SQLite index path")
    search.add_argument("--query", required=True, help="Query text")
    search.add_argument("--top-k", type=int, default=5, help="Maximum passages to return (default: 5)")
    search.add_argument("--scope", help="Search only this path relative to the indexed directory")
    search.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid",
                        help="Search strategy (default: hybrid; keyword does not load the model)")
    search.add_argument("--json", action="store_true", help="Return a JSON array")
    args = parser.parse_args()

    try:
        if args.command == "index":
            print(json.dumps(build_index(args.input, args.index, E5Embedder())))
        elif args.command == "status":
            print(json.dumps(index_status(args.index), ensure_ascii=False))
        else:
            hits = search_index(args.index, args.query, args.top_k, E5Embedder(),
                                scope=args.scope, mode=args.mode)
            if args.json:
                print(json.dumps(hits, ensure_ascii=False, allow_nan=False))
            elif not hits:
                print("No passages in the index.")
            else:
                for hit in hits:
                    print(f"{hit['source']}:{hit['start_line']}-{hit['end_line']} ({hit['score']:.3f})")
                    print(hit["text"], end="\n\n")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    return 0

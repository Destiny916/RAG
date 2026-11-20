import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .documents import DOCUMENT_KINDS
from .embedding import E5Embedder
from .index import assess_candidates, build_index, index_status, search_index


def main() -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Index and search local documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Rebuild the document index")
    index.add_argument("--input", type=Path, required=True, help="Directory of UTF-8 Markdown and text files")
    index.add_argument("--index", type=Path, required=True, help="SQLite output path")
    index.add_argument("--announce-to", type=Path, metavar="AGENTS.md",
                       help="Report published changes in this AGENTS.md after a successful build")

    status = commands.add_parser("status", help="Show index size and available scopes as JSON")
    status.add_argument("--index", type=Path, required=True, help="SQLite index path")

    search = commands.add_parser("search", help="Return the most relevant source passages")
    search.add_argument("--index", type=Path, required=True, help="SQLite index path")
    search.add_argument("--query", action="append", required=True, help="Query text; may be repeated with --json")
    search.add_argument("--top-k", type=int, default=5, help="Maximum passages to return (default: 5)")
    search.add_argument("--scope", help="Search only this path relative to the indexed directory")
    search.add_argument("--kind", choices=(*DOCUMENT_KINDS, "all"), default="guide",
                        help="Document kind to search (default: guide)")
    search.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid",
                        help="Search strategy (default: hybrid; keyword does not load the model)")
    search.add_argument("--json", action="store_true", help="Return a JSON array")
    search.add_argument("--evidence", action="store_true", help="Report whether candidates have lexical support")
    search.add_argument("--intent", choices=("reference", "live"), default="reference",
                        help="With --evidence: live facts require observations of the target")

    review = commands.add_parser("review", help="Submit, inspect, or resolve local intake drafts")
    review.add_argument("--root", type=Path, default=Path.cwd(), help="RAG repository root (default: current directory)")
    actions = review.add_subparsers(dest="review_action", required=True)
    listing = actions.add_parser("list")
    listing.add_argument("--state", action="append", help="Filter by state; may be repeated")
    actions.add_parser("submit", help="Read one JSON draft from stdin")
    actions.add_parser("show").add_argument("id")
    actions.add_parser("reject").add_argument("id")
    approval = actions.add_parser("approve", help="Publish a human-approved preview")
    approval.add_argument("id")
    approval.add_argument("--digest", required=True, help="Digest printed by review show")
    approval.add_argument("--announce-to", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "index":
            if args.announce_to:
                from .publication import publish_index

                result = publish_index(args.input, args.index, E5Embedder(), args.announce_to)
            else:
                result = build_index(args.input, args.index, E5Embedder())
            print(json.dumps(result))
        elif args.command == "status":
            print(json.dumps(index_status(args.index), ensure_ascii=False))
        elif args.command == "review":
            from . import reviews
            from .intake import submit

            root = args.root.resolve()
            if args.review_action == "list":
                result = [entry for entry in reviews.list_reviews(root)
                          if not args.state or entry["state"] in args.state]
            elif args.review_action == "submit":
                result = {"id": submit(json.loads(sys.stdin.read(70000)), root)}
            elif args.review_action == "show":
                result = reviews.show(root, args.id)
            elif args.review_action == "reject":
                result = reviews.reject(root, args.id)
            else:
                result = reviews.approve(root, args.id, args.digest, args.announce_to)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            multiple = len(args.query) > 1
            if multiple and not args.json:
                raise ValueError("repeated --query requires --json")
            if args.intent == "live" and not args.evidence:
                raise ValueError("--intent live requires --evidence")
            model = E5Embedder()
            results = []
            for query in args.query:
                hits = search_index(args.index, query, args.top_k, model,
                                    scope=args.scope, mode=args.mode, kind=args.kind)
                results.append(assess_candidates(query, hits, live=args.intent == "live")
                               if args.evidence else hits)
            if args.json or args.evidence:
                print(json.dumps(results if multiple else results[0], ensure_ascii=False, allow_nan=False))
            elif not hits:
                print("No matching passages in the selected scope and document kind.")
            else:
                for hit in hits:
                    print(f"{hit['source']}:{hit['start_line']}-{hit['end_line']} "
                          f"[{hit['kind']}] ({hit['score']:.3f})")
                    print(hit["text"], end="\n\n")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    return 0

import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np

from .documents import (
    DOCUMENT_KINDS, document_kind, read_chunks, read_kind_rules, searchable_text, split_document,
)
from .embedding import Embedder
from .lexical import match_expression, technical_terms, terms

APPLICATION_ID = 0x52414731
SCHEMA_VERSION = 3
VECTOR_DTYPE = np.dtype("<f4")
SearchMode = Literal["hybrid", "keyword", "semantic"]
RANK_OFFSET = 20


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True)


def _check_index(db: sqlite3.Connection, path: Path) -> None:
    try:
        application_id = db.execute("PRAGMA application_id").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        if exc.sqlite_errorcode != sqlite3.SQLITE_NOTADB:
            raise
        raise ValueError(f"not a RAG index: {path}") from exc
    if application_id != APPLICATION_ID:
        raise ValueError(f"not a RAG index: {path}")


def _open_existing(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError(f"index does not exist: {path}; run rag index first")
    db = _connect_readonly(path)
    try:
        _check_index(db, path)
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unsupported index version; rebuild with rag index")
    except BaseException:
        db.close()
        raise
    return db


def _cached_embeddings(path: Path, embedder: Embedder) -> dict[bytes, bytes]:
    if path.is_symlink():
        raise ValueError(f"index output must not be a symbolic link: {path}")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"index output is not a regular file: {path}")
    with closing(_connect_readonly(path)) as db:
        _check_index(db, path)
        # v3 adds source metadata without changing the embedding input or storage.
        if db.execute("PRAGMA user_version").fetchone()[0] not in (2, SCHEMA_VERSION):
            return {}
        name, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        if (name, revision) != (embedder.model_id, embedder.revision):
            return {}
        return {key: vector for key, vector in db.execute("SELECT embedding_key, vector FROM chunks")
                if len(vector) == dimension * VECTOR_DTYPE.itemsize}


def build_index(root: Path, path: Path, embedder: Embedder) -> dict:
    cache = _cached_embeddings(path, embedder)
    rules = read_kind_rules(root)
    chunks = list(read_chunks(root))
    kinds = {source: document_kind(source, rules) for source in {chunk.source for chunk in chunks}}
    keys, pending = [], {}
    for chunk in chunks:
        contents = chunk.embedding_text
        key = sha256(contents.encode("utf-8")).digest()
        keys.append(key)
        if key not in cache:
            pending[key] = contents
    if pending:
        vectors = np.asarray(embedder.encode(list(pending.values()), kind="passage"), dtype=VECTOR_DTYPE)
        if (vectors.ndim != 2 or vectors.shape[0] != len(pending)
                or not vectors.shape[1] or not np.isfinite(vectors).all()):
            raise ValueError("embedding model returned invalid passage vectors")
        cache.update((key, vector.tobytes()) for key, vector in zip(pending, vectors, strict=True))
    dimension = len(cache[keys[0]]) // VECTOR_DTYPE.itemsize if keys else 0
    if any(len(cache[key]) != dimension * VECTOR_DTYPE.itemsize for key in keys):
        raise ValueError("embedding vector size changed; rebuild into a new index file")

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".rag-build-", delete=False) as temp:
        staging = Path(temp.name)
    try:
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("CREATE TABLE model (name TEXT, revision TEXT, dimension INTEGER)")
            db.execute("INSERT INTO model VALUES (?, ?, ?)", (
                embedder.model_id, embedder.revision, dimension,
            ))
            db.execute("""CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, title TEXT,
                section TEXT, table_header TEXT,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, embedding_key BLOB NOT NULL, vector BLOB NOT NULL
            )""")
            db.executemany("""INSERT INTO chunks
                (source, kind, title, section, table_header, start_line, end_line, text, embedding_key, vector)
                VALUES (:source, :kind, :title, :section, :table_header, :start_line, :end_line, :text, :key, :vector)""", (
                {**asdict(chunk), "kind": kinds[chunk.source], "key": key, "vector": cache[key]}
                for chunk, key in zip(chunks, keys, strict=True)
            ))
            db.execute("CREATE VIRTUAL TABLE keywords USING fts5(body, heading, source)")
            db.executemany("INSERT INTO keywords (rowid, body, heading, source) VALUES (?, ?, ?, ?)", (
                (i, " ".join(terms(searchable_text(chunk.text))),
                 " ".join(terms(searchable_text(chunk.section or ""))), " ".join(terms(chunk.source)))
                for i, chunk in enumerate(chunks, start=1)
            ))
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)

    return {"documents": len(kinds), "chunks": len(chunks)}


def index_status(path: Path) -> dict:
    with closing(_open_existing(path)) as db:
        name, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        sources = db.execute("SELECT source, kind, count(*) FROM chunks GROUP BY source, kind").fetchall()
    scopes = Counter(source.partition("/")[0] for source, _, _ in sources)
    kinds = Counter(kind for _, kind, _ in sources)
    return {"schema": SCHEMA_VERSION, "model": name, "revision": revision, "dimension": dimension,
            "documents": len(sources), "chunks": sum(count for _, _, count in sources),
            "scopes": dict(sorted(scopes.items())), "kinds": dict(sorted(kinds.items()))}


def source_manifest(path: Path) -> dict[str, list[str]]:
    manifest = {}
    with closing(_open_existing(path)) as db:
        rows = db.execute("""SELECT source, kind, title, section, start_line, end_line, text
                             FROM chunks ORDER BY source, id""")
        for source, chunks in groupby(rows, key=lambda row: row[0]):
            digest = sha256()
            for _, kind, *content in chunks:
                digest.update(json.dumps(content).encode())
                digest.update(b"\n")
            manifest[source] = [kind, digest.hexdigest()]
    return manifest


def read_indexed_source(path: Path, root: Path, source: str, kind: str, *,
                        start_line: int, end_line: int) -> dict:
    if "\\" in source or any(part in {"", ".", ".."} for part in source.split("/")):
        raise ValueError("source must be a path relative to the knowledge directory")
    if kind not in DOCUMENT_KINDS:
        raise ValueError("unknown document kind")
    if not 1 <= start_line <= end_line < start_line + 60:
        raise ValueError("request between 1 and 60 lines, with 1-based numbering")
    with closing(_open_existing(path)) as db:
        indexed = db.execute(
            "SELECT start_line, end_line, text FROM chunks WHERE source = ? AND kind = ? ORDER BY id",
            (source, kind),
        ).fetchall()
    if not indexed:
        raise ValueError("source is not in the requested document kind")

    root = root.resolve(strict=True)
    file = root / source
    if file.resolve(strict=True) != file or not file.is_file():
        raise ValueError("source is not a regular file in the knowledge directory")
    text = file.read_text(encoding="utf-8-sig")
    current = [(chunk.start_line, chunk.end_line, chunk.text) for chunk in split_document(source, text)]
    if current != indexed or document_kind(source, read_kind_rules(root)) != kind:
        raise ValueError("source changed since indexing; rebuild the index before citing it")
    lines = text.removesuffix("\n").split("\n")
    if start_line > len(lines):
        raise ValueError("start_line is past the end of the source")
    end_line = min(end_line, len(lines))
    excerpt = "\n".join(lines[start_line - 1:end_line])
    if len(excerpt) > 12000:
        raise ValueError("excerpt exceeds 12000 characters; narrow the line range or use the search fragment")
    return {"source": source, "kind": kind, "start_line": start_line, "end_line": end_line, "text": excerpt}


def search_index(path: Path, query: str, top_k: int, embedder: Embedder, *,
                 scope: str | None = None, mode: SearchMode = "hybrid", kind: str = "guide") -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    if mode not in {"hybrid", "keyword", "semantic"}:
        raise ValueError("unknown search mode")
    if kind != "all" and kind not in DOCUMENT_KINDS:
        raise ValueError("unknown document kind")
    clause, params = ("1", ()) if kind == "all" else ("c.kind = ?", (kind,))
    if scope is not None:
        scope = scope.strip().rstrip("/")
        if not scope or any(part in {"", ".", ".."} for part in scope.split("/")):
            raise ValueError("scope must be a path relative to the indexed directory")
        clause += " AND (c.source = ? OR substr(c.source, 1, ?) = ?)"
        params += (scope, len(scope) + 1, scope + "/")
    limit = max(top_k * 4, 40)
    with closing(_open_existing(path)) as db:
        model, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        if mode != "keyword" and (model, revision) != (embedder.model_id, embedder.revision):
            raise ValueError("index model has changed; rebuild with rag index")
        db.row_factory = sqlite3.Row
        keyword_rows, semantic_rows = [], []
        expression = match_expression(query) if mode != "semantic" else ""
        if expression:
            keyword_rows = db.execute(f"""SELECT c.id, c.source, c.kind, c.title, c.section, c.table_header,
                c.start_line, c.end_line, c.text FROM keywords JOIN chunks c ON c.id = keywords.rowid
                WHERE keywords MATCH ? AND {clause} ORDER BY bm25(keywords, 5.0, 1.6, 0.8), c.id LIMIT ?""",
                (expression, *params, limit)).fetchall()
        if mode != "keyword":
            semantic_rows = db.execute(f"""SELECT id, source, kind, title, section, table_header,
                start_line, end_line, text, vector FROM chunks c WHERE {clause} ORDER BY id""", params).fetchall()

    query_terms = technical_terms(query)
    exact: dict[int, float] = {}
    items: dict[int, dict] = {}
    scores: dict[int, float] = {}
    if semantic_rows:
        query_vector = np.asarray(embedder.encode([query], kind="query"), dtype=VECTOR_DTYPE)
        if query_vector.shape != (1, dimension) or not np.isfinite(query_vector).all():
            raise ValueError("embedding model returned an invalid query vector")
        vectors = np.frombuffer(b"".join(row["vector"] for row in semantic_rows), dtype=VECTOR_DTYPE)
        similarity = vectors.reshape((len(semantic_rows), dimension)) @ query_vector[0]
        for rank, i in enumerate(np.argsort(-similarity, kind="stable")[:limit], 1):
            item = dict(semantic_rows[i])
            item.pop("vector")
            item_id = item.pop("id")
            item["semantic_score"] = float(similarity[i])
            items[item_id] = item
            scores[item_id] = item["semantic_score"] if mode == "semantic" else 1 / (RANK_OFFSET + rank)

    if query_terms:
        for row in keyword_rows:
            identifiers = technical_terms(searchable_text(row["text"]))
            exact[row["id"]] = len(query_terms & identifiers) / len(query_terms)
        keyword_rows.sort(key=lambda row: -exact[row["id"]])
    for rank, row in enumerate(keyword_rows, 1):
        item_id = row["id"]
        if item_id not in items:
            items[item_id] = dict(row)
            items[item_id].pop("id")
        items[item_id]["keyword_rank"] = rank
        scores[item_id] = scores.get(item_id, 0) + 1 / (RANK_OFFSET + rank) + 0.025 * exact.get(item_id, 0)

    return [{**items[i], "score": scores[i]} for i in sorted(scores, key=lambda i: (-scores[i], i))[:top_k]]


def assess_candidates(query: str, hits: list[dict], *, live: bool = False) -> dict:
    # Rank is relative: lexical corroboration is a conservative signal, not a proof.
    query_terms = set(terms(query, query=True))
    identifiers = technical_terms(query)
    corroborated = []
    for hit in hits:
        context = searchable_text("\n".join(filter(None, (hit["section"], hit["text"]))))
        overlap = query_terms & set(terms(context, query=True))
        exact = identifiers & technical_terms(context)
        # Count independent matches, not all the overlapping n-grams of one phrase.
        matching = sorted((bytes.fromhex(term[1:]).decode() for term in overlap),
                          key=lambda word: (-len(word), word))
        distinct = []
        for term in matching:
            if len(term) > 1 and not any(term in other or other in term or
                                        (min(len(term), len(other)) >= 2 and
                                         (term[-2:] == other[:2] or other[-2:] == term[:2]))
                                        for other in distinct):
                distinct.append(term)
        if exact or len(distinct) >= 2:
            corroborated.append(hit)
    state = "live_required" if live else ("candidates" if corroborated else "insufficient")
    return {"query": query, "status": state, "matches": corroborated}

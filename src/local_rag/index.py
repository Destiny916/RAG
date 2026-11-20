import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np

from .documents import read_chunks, searchable_text
from .embedding import Embedder
from .lexical import match_expression, technical_terms, terms

APPLICATION_ID = 0x52414731
SCHEMA_VERSION = 2
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
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            return {}
        name, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        if (name, revision) != (embedder.model_id, embedder.revision):
            return {}
        return {key: vector for key, vector in db.execute("SELECT embedding_key, vector FROM chunks")
                if len(vector) == dimension * VECTOR_DTYPE.itemsize}


def build_index(root: Path, path: Path, embedder: Embedder) -> dict:
    cache = _cached_embeddings(path, embedder)
    chunks = list(read_chunks(root))
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
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, title TEXT, section TEXT, table_header TEXT,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, embedding_key BLOB NOT NULL, vector BLOB NOT NULL
            )""")
            db.executemany("""INSERT INTO chunks
                (source, title, section, table_header, start_line, end_line, text, embedding_key, vector)
                VALUES (:source, :title, :section, :table_header, :start_line, :end_line, :text, :key, :vector)""", (
                {**asdict(chunk), "key": key, "vector": cache[key]}
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

    return {"documents": len({chunk.source for chunk in chunks}), "chunks": len(chunks)}


def index_status(path: Path) -> dict:
    with closing(_open_existing(path)) as db:
        name, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        sources = db.execute("SELECT source, count(*) FROM chunks GROUP BY source").fetchall()
    scopes = Counter(source.partition("/")[0] for source, _ in sources)
    return {"schema": SCHEMA_VERSION, "model": name, "revision": revision, "dimension": dimension,
            "documents": len(sources), "chunks": sum(count for _, count in sources),
            "scopes": dict(sorted(scopes.items()))}


def search_index(path: Path, query: str, top_k: int, embedder: Embedder, *,
                 scope: str | None = None, mode: SearchMode = "hybrid") -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    if mode not in {"hybrid", "keyword", "semantic"}:
        raise ValueError("unknown search mode")
    clause, params = "1", ()
    if scope is not None:
        scope = scope.strip().rstrip("/")
        if not scope or any(part in {"", ".", ".."} for part in scope.split("/")):
            raise ValueError("scope must be a path relative to the indexed directory")
        clause = "(c.source = ? OR substr(c.source, 1, ?) = ?)"
        params = (scope, len(scope) + 1, scope + "/")
    limit = max(top_k * 4, 40)
    with closing(_open_existing(path)) as db:
        model, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        if mode != "keyword" and (model, revision) != (embedder.model_id, embedder.revision):
            raise ValueError("index model has changed; rebuild with rag index")
        db.row_factory = sqlite3.Row
        keyword_rows, semantic_rows = [], []
        expression = match_expression(query) if mode != "semantic" else ""
        if expression:
            keyword_rows = db.execute(f"""SELECT c.id, c.source, c.title, c.section, c.table_header,
                c.start_line, c.end_line, c.text FROM keywords JOIN chunks c ON c.id = keywords.rowid
                WHERE keywords MATCH ? AND {clause} ORDER BY bm25(keywords, 5.0, 1.6, 0.8), c.id LIMIT ?""",
                (expression, *params, limit)).fetchall()
        if mode != "keyword":
            semantic_rows = db.execute(f"""SELECT id, source, title, section, table_header,
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

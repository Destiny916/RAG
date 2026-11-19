import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .documents import read_chunks
from .embedding import Embedder

APPLICATION_ID = 0x52414731
SCHEMA_VERSION = 1
VECTOR_DTYPE = np.dtype("<f4")


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


def build_index(root: Path, path: Path, embedder: Embedder) -> dict:
    if path.is_symlink():
        raise ValueError(f"index output must not be a symbolic link: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"index output is not a regular file: {path}")
        with closing(_connect_readonly(path)) as original:
            _check_index(original, path)

    chunks = list(read_chunks(root))
    if chunks:
        vectors = np.asarray(
            embedder.encode([chunk.embedding_text for chunk in chunks], kind="passage"),
            dtype=VECTOR_DTYPE,
        )
        if (vectors.ndim != 2 or vectors.shape[0] != len(chunks)
                or not vectors.shape[1] or not np.isfinite(vectors).all()):
            raise ValueError("embedding model returned invalid passage vectors")
    else:
        vectors = np.empty((0, 0), dtype=VECTOR_DTYPE)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".rag-build-", delete=False) as temp:
        staging = Path(temp.name)
    try:
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("CREATE TABLE model (name TEXT, revision TEXT, dimension INTEGER)")
            db.execute("INSERT INTO model VALUES (?, ?, ?)", (
                embedder.model_id, embedder.revision, vectors.shape[1],
            ))
            db.execute("""CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, title TEXT,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                text TEXT NOT NULL, vector BLOB NOT NULL
            )""")
            db.executemany("""INSERT INTO chunks (source, title, start_line, end_line, text, vector)
                VALUES (:source, :title, :start_line, :end_line, :text, :vector)""", (
                {**asdict(chunk), "vector": vector.tobytes()} for chunk, vector in zip(chunks, vectors)
            ))
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)

    return {"documents": len({chunk.source for chunk in chunks}), "chunks": len(chunks)}


def search_index(path: Path, query: str, top_k: int, embedder: Embedder) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    if not path.is_file():
        raise ValueError(f"index does not exist: {path}; run rag index first")

    with closing(_connect_readonly(path)) as db:
        _check_index(db, path)
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unsupported index version; rebuild with rag index")
        model, revision, dimension = db.execute("SELECT name, revision, dimension FROM model").fetchone()
        if (model, revision) != (embedder.model_id, embedder.revision):
            raise ValueError("index model has changed; rebuild with rag index")
        db.row_factory = sqlite3.Row
        rows = db.execute("""SELECT source, title, start_line, end_line, text, vector
            FROM chunks ORDER BY id""").fetchall()

    if not rows:
        return []
    query_vector = np.asarray(embedder.encode([query], kind="query"), dtype=VECTOR_DTYPE)
    if query_vector.shape != (1, dimension) or not np.isfinite(query_vector).all():
        raise ValueError("embedding model returned an invalid query vector")
    vectors = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=VECTOR_DTYPE)
    scores = vectors.reshape((len(rows), dimension)) @ query_vector[0]
    hits = []
    for i in np.argsort(-scores, kind="stable")[:top_k]:
        hit = dict(rows[i])
        del hit["vector"]
        hit["score"] = float(scores[i])
        hits.append(hit)
    return hits

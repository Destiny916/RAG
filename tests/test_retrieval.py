import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from local_rag.documents import CHUNK_CHARS, split_document
from local_rag.embedding import E5Embedder
from local_rag.index import build_index, search_index

FIXTURES = Path(__file__).parent / "fixtures"


class TestEmbedder:
    __test__ = False
    model_id = "fixture"
    revision = "test"

    def encode(self, texts, *, kind):
        return np.array([
            [1, 0] if kind == "query" or "ICMP" in text else [0, 1]
            for text in texts
        ], dtype=np.float32)


def test_markdown_preserves_source_lines():
    text = "# C#\n\n```md\n# not a heading\n```\n\nOther title\n-----------\n\nMore text\n"
    chunks = list(split_document("guide.md", text))
    assert [(c.title, c.start_line, c.end_line) for c in chunks] == [
        ("C#", 1, 5), ("Other title", 7, 10),
    ]
    for chunk in chunks:
        assert chunk.text == "\n".join(text.splitlines()[chunk.start_line - 1:chunk.end_line])


def test_long_line_keeps_its_citation():
    line = "ab" * CHUNK_CHARS
    chunks = list(split_document("guide.txt", f"\n{line}\n"))
    assert "".join(chunk.text for chunk in chunks) == line
    assert all(chunk.start_line == chunk.end_line == 2 for chunk in chunks)
    assert all(len(chunk.text) <= CHUNK_CHARS for chunk in chunks)


def test_rebuild_drops_deleted_files_and_preserves_old_index_on_error(tmp_path):
    root = tmp_path / "docs"
    shutil.copytree(FIXTURES, root)
    index = tmp_path / "index #1.sqlite"
    embedder = TestEmbedder()
    assert build_index(root, index, embedder) == {"documents": 6, "chunks": 6}
    (root / "connection.md").unlink()
    bad_file = root / "invalid.txt"
    bad_file.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        build_index(root, index, embedder)
    assert search_index(index, "connection", 1, embedder)[0]["source"] == "connection.md"

    bad_file.unlink()
    assert build_index(root, index, embedder) == {"documents": 5, "chunks": 5}
    assert "connection.md" not in {hit["source"] for hit in search_index(index, "connection", 10, embedder)}


def test_missing_index_and_empty_index_do_not_load_model(tmp_path):
    embedder = E5Embedder()
    index = tmp_path / "empty.sqlite"
    with pytest.raises(ValueError, match="run rag index first"):
        search_index(index, "test", 1, embedder)
    assert build_index(tmp_path, index, embedder) == {"documents": 0, "chunks": 0}
    assert search_index(index, "test", 1, embedder) == []
    assert "model" not in vars(embedder)


def test_unrelated_output_file_is_preserved(tmp_path):
    output = tmp_path / "notes.txt"
    output.write_bytes(b"keep this file")
    with pytest.raises(ValueError, match="not a RAG index"):
        build_index(FIXTURES, output, TestEmbedder())
    assert output.read_bytes() == b"keep this file"


@pytest.mark.model
def test_real_cli_retrieval(tmp_path):
    rag = Path(sys.executable).with_name("rag")
    index = tmp_path / "e5.sqlite"

    def run(*args):
        result = subprocess.run([rag, *args, "--index", index], capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert run("index", "--input", FIXTURES) == {"documents": 6, "chunks": 6}
    for query in ("怎样确认网络已经通了？", "How do I check that the network is reachable?"):
        hits = run("search", "--query", query, "--top-k", "5", "--json")
        assert len(hits) == 5
        assert hits[0]["source"] == "connection.md"
        assert hits[0]["title"] == "测试环境连通性"
        assert (hits[0]["start_line"], hits[0]["end_line"]) == (1, 3)
        assert "丢包率为零" in hits[0]["text"]
        assert hits == sorted(hits, key=lambda hit: hit["score"], reverse=True)

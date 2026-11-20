import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from local_rag.documents import CHUNK_CHARS, split_document
from local_rag.embedding import E5Embedder
from local_rag.index import build_index, index_status, search_index


class StubEmbedder:
    model_id = "fixture"
    revision = "test"

    def encode(self, texts, *, kind):
        return np.array([
            [1, 0] if kind == "query" or "钥匙" in text else [0, 1]
            for text in texts
        ], dtype=np.float32)


@pytest.fixture
def sample_docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in {
        "target.md": "# 样本标签\n\n一串钥匙放在窗边的方盒里。\n",
        "a.txt": "三头大象在草地上散步。\n",
        "b.txt": "远处的山顶覆盖着白雪。\n",
        "c.txt": "盘子上摆着两块蛋糕。\n",
        "d.txt": "弦乐四重奏即将开始。\n",
        "e.txt": "这张纸是蓝色的。\n",
    }.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_markdown_preserves_source_lines():
    text = "# C#\n\n```md\n# not a heading\n```\n\nOther title\n-----------\n\nMore text\n"
    chunks = list(split_document("guide.md", text))
    assert [(c.title, c.start_line, c.end_line) for c in chunks] == [
        ("C#", 1, 1), ("C#", 3, 5), ("Other title", 7, 10),
    ]
    for chunk in chunks:
        assert chunk.text == "\n".join(text.splitlines()[chunk.start_line - 1:chunk.end_line])


def test_technical_evidence_keeps_section_and_citable_block():
    text = """# 操作手册
## 管理任务
### 启动命令

```bash
runner start \\
  --data-root /tmp/sample \\
  --format text
```

下面是另一份配置的参数。

| 参数 | 用途 |
| --- | --- |
| `--data-root` | 数据目录 |
| `--format` | 输出格式 |
"""
    chunks = list(split_document("guide.md", text))
    command = next(c for c in chunks if "runner start" in c.text)
    assert command.section == "操作手册 > 管理任务 > 启动命令"
    assert command.text == "\n".join(text.splitlines()[4:9])
    table = next(c for c in chunks if "数据目录" in c.text)
    assert table.table_header == "| 参数 | 用途 |"
    assert table.text == "\n".join(text.splitlines()[12:16])


def test_long_line_keeps_its_citation():
    line = "ab" * CHUNK_CHARS
    chunks = list(split_document("guide.txt", f"\n{line}\n"))
    assert "".join(chunk.text for chunk in chunks) == line
    assert all(chunk.start_line == chunk.end_line == 2 for chunk in chunks)
    assert all(len(chunk.text) <= CHUNK_CHARS for chunk in chunks)


def test_rebuild_drops_deleted_files_and_preserves_old_index_on_error(tmp_path, sample_docs):
    index = tmp_path / "index #1.sqlite"
    embedder = StubEmbedder()
    assert build_index(sample_docs, index, embedder) == {"documents": 6, "chunks": 6}
    (sample_docs / "target.md").unlink()
    bad_file = sample_docs / "invalid.txt"
    bad_file.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        build_index(sample_docs, index, embedder)
    assert search_index(index, "keys", 1, embedder)[0]["source"] == "target.md"

    bad_file.unlink()
    assert build_index(sample_docs, index, embedder) == {"documents": 5, "chunks": 5}
    assert "target.md" not in {hit["source"] for hit in search_index(index, "keys", 10, embedder)}


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
        build_index(tmp_path, output, StubEmbedder())
    assert output.read_bytes() == b"keep this file"


def test_keyword_prefer_exact_identifier_and_respect_scope(tmp_path):
    root = tmp_path / "docs"
    alpha = root / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "guide.md").write_text("# Export\n\nUse `--out-dir` to choose the export location.")
    (alpha / "notes.txt").write_text("The out directory is where the export goes.")
    beta = root / "alpha_legacy"
    beta.mkdir()
    (beta / "guide.md").write_text("# Export\n\nUse `--out-dir` for historical reports.")
    index = tmp_path / "index.sqlite"
    build_index(root, index, StubEmbedder())
    embedder = E5Embedder()

    hits = search_index(index, "--out-dir", 5, embedder, scope="alpha", mode="keyword")
    assert hits[0]["source"] == "alpha/guide.md"
    assert all(hit["source"].startswith("alpha/") for hit in hits)
    assert search_index(index, "mismatched ? ' literal", 5, embedder, mode="keyword") == []
    assert search_index(index, "--out-dir", 5, embedder, scope="bad-scope", mode="keyword") == []
    assert "model" not in vars(embedder)
    assert index_status(index)["scopes"] == {"alpha": 2, "alpha_legacy": 1}


def test_rebuild_reuses_vectors_and_refreshes_citations(tmp_path, sample_docs):
    class RecordingEmbedder(StubEmbedder):
        def __init__(self):
            self.passage_batches = []

        def encode(self, texts, *, kind):
            if kind == "passage":
                self.passage_batches.append(tuple(texts))
            return super().encode(texts, kind=kind)

    embedder = RecordingEmbedder()
    index = tmp_path / "index.sqlite"
    build_index(sample_docs, index, embedder)
    original_file = sample_docs / "target.md"
    original_file.write_text("\n\n" + original_file.read_text(), encoding="utf-8")
    (sample_docs / "a.txt").write_text("新增一条核对记录。", encoding="utf-8")
    build_index(sample_docs, index, embedder)
    assert embedder.passage_batches[-1] == ("新增一条核对记录。",)
    hits = search_index(index, "钥匙", 1, embedder, mode="keyword")
    assert (hits[0]["source"], hits[0]["start_line"]) == ("target.md", 3)


def test_script_query_returns_command_and_its_adjacent_parameter_table(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "train.md").write_text(
        "# 训练\n\n```bash\nbash scripts/run_job.sh\n```\n\n"
        "| 参数 | 用途 |\n| --- | --- |\n| `JOB_DATA_ROOT` | 数据目录 |\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.sqlite"
    build_index(root, index, StubEmbedder())
    hit = search_index(index, "run_job.sh JOB_DATA_ROOT", 1, E5Embedder(), mode="keyword")[0]
    assert "bash scripts/run_job.sh" in hit["text"]
    assert "JOB_DATA_ROOT" in hit["text"]
    assert (hit["start_line"], hit["end_line"]) == (3, 9)


@pytest.mark.model
def test_real_cli_retrieval(tmp_path, sample_docs):
    rag = Path(sys.executable).with_name("rag")
    index = tmp_path / "e5.sqlite"

    def run(*args):
        result = subprocess.run([rag, *args, "--index", index], capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert run("index", "--input", sample_docs) == {"documents": 6, "chunks": 6}
    for query in ("到哪里能找到那串钥匙？", "Where can I find the keys?"):
        hits = run("search", "--query", query, "--top-k", "5", "--json")
        assert len(hits) == 5
        assert hits[0]["source"] == "target.md"
        assert hits[0]["title"] == "样本标签"
        assert (hits[0]["start_line"], hits[0]["end_line"]) == (1, 3)
        assert "窗边的方盒" in hits[0]["text"]
        assert hits == sorted(hits, key=lambda hit: hit["score"], reverse=True)
    assert run("status")["documents"] == 6
    hits = run("search", "--query", "钥匙", "--scope", "target.md", "--mode", "keyword", "--json")
    assert len(hits) == 1 and hits[0]["source"] == "target.md"

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from local_rag import intake, publication, reviews
from local_rag.documents import CHUNK_CHARS, split_document
from local_rag.embedding import E5Embedder
from local_rag.index import build_index, index_status, read_indexed_source, search_index


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


def test_read_source_requires_indexed_kind_and_current_lines(tmp_path, sample_docs):
    index = tmp_path / "index.sqlite"
    build_index(sample_docs, index, StubEmbedder())
    result = read_indexed_source(index, sample_docs, "target.md", "guide", start_line=3, end_line=5)
    assert result["text"] == "一串钥匙放在窗边的方盒里。"
    assert result["start_line"] == result["end_line"] == 3
    with pytest.raises(ValueError, match="requested document kind"):
        read_indexed_source(index, sample_docs, "target.md", "site", start_line=1, end_line=1)
    with pytest.raises(ValueError, match="relative"):
        read_indexed_source(index, sample_docs, "../target.md", "guide", start_line=1, end_line=1)
    source = sample_docs / "target.md"
    source.write_text("\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="source changed since indexing"):
        read_indexed_source(index, sample_docs, "target.md", "guide", start_line=3, end_line=3)


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


def test_publication_updates_agent_inventory_after_success(tmp_path, sample_docs):
    index = tmp_path / "index.sqlite"
    agents = tmp_path / "AGENTS.md"
    original = "# User rules\n\nContinue to follow these rules.\n"
    agents.write_text(original)
    embedder = StubEmbedder()
    build_index(sample_docs, index, embedder)
    assert publication.publish_index(sample_docs, index, embedder, agents)["announcement"] == {
        "added": 0, "updated": 0, "removed": 0,
    }
    original_report = agents.read_text()

    new_dir = sample_docs / "new-project"
    new_dir.mkdir()
    (new_dir / "guide.md").write_text("Document body stays out of the announcement.")
    (sample_docs / "target.md").write_text("Updated document body.")
    (sample_docs / "e.txt").unlink()
    invalid = sample_docs / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        publication.publish_index(sample_docs, index, embedder, agents)
    assert agents.read_text() == original_report
    assert index_status(index)["documents"] == 6

    invalid.unlink()
    assert publication.publish_index(sample_docs, index, embedder, agents)["announcement"] == {
        "added": 1, "updated": 1, "removed": 1,
    }
    report = agents.read_text()
    assert report.startswith(original)
    for name in ("`new-project`", "`new-project/guide.md`", "`target.md`", "`e.txt`"):
        assert name in report
    assert "Document body stays" not in report
    publication.publish_index(sample_docs, index, embedder, agents)
    assert agents.read_text() == report


def test_publication_retries_notification_after_index_was_updated(tmp_path, sample_docs, monkeypatch):
    index = tmp_path / "index.sqlite"
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# User rules\n")
    embedder = StubEmbedder()
    build_index(sample_docs, index, embedder)
    (sample_docs / "new.md").write_text("A new document.")
    write_atomic = publication._write_atomic

    def fail_agents_write(path, *args):
        if path == agents:
            raise OSError("announcement cannot be written")
        write_atomic(path, *args)

    with monkeypatch.context() as context:
        context.setattr(publication, "_write_atomic", fail_agents_write)
        with pytest.raises(OSError, match="announcement cannot be written"):
            publication.publish_index(sample_docs, index, embedder, agents)
    assert index_status(index)["documents"] == 7
    assert agents.read_text() == "# User rules\n"
    assert publication.publish_index(sample_docs, index, embedder, agents)["announcement"]["added"] == 1
    assert "`new.md`" in agents.read_text()


def test_review_approves_exact_diff_and_recovers_after_build_failure(tmp_path, monkeypatch):
    root = tmp_path
    knowledge = root / "knowledge"
    knowledge.mkdir()
    agents = root / "AGENTS.md"
    agents.write_text("# User rules\n")
    draft = {
        "action": "new_directory", "target": "alpha/guide.md", "kind": "guide",
        "summary": "A reusable example", "source": "example.py:1",
        "verified_on": date.today().isoformat(), "scope": "alpha",
        "content": "# Example\n\nA verified example.\n",
    }
    token = intake.submit(draft, root)
    assert intake.submit(draft, root) == token
    assert [entry["state"] for entry in reviews.list_reviews(root)] == ["draft"]
    preview = reviews.show(root, token[:12])
    assert "+A verified example." in preview["diff"]
    with pytest.raises(ValueError, match="digest changed"):
        reviews.approve(root, token, "wrong", agents, StubEmbedder())

    def broken_build(*args):
        raise RuntimeError("build failed")

    with monkeypatch.context() as context:
        context.setattr(publication, "build_index", broken_build)
        with pytest.raises(RuntimeError, match="build failed"):
            reviews.approve(root, token, preview["digest"], agents, StubEmbedder())
    assert not (knowledge / "alpha").exists()
    assert "alpha" not in agents.read_text()
    assert reviews.show(root, token)["state"] == "publish_pending"
    result = reviews.approve(root, token, preview["digest"], agents, StubEmbedder())
    assert result["state"] == "published"
    assert (knowledge / draft["target"]).read_text() == draft["content"]
    assert "`alpha/guide.md`" in agents.read_text()
    assert index_status(root / ".rag" / "project.sqlite")["documents"] == 1

    changed_draft = {**draft, "action": "update", "content": "# Different\n"}
    changed = intake.submit(changed_draft, root)
    digest = reviews.show(root, changed)["digest"]
    (knowledge / draft["target"]).write_text("# Concurrent change\n")
    with pytest.raises(ValueError, match="changed since this draft"):
        reviews.approve(root, changed, digest, agents, StubEmbedder())
    assert reviews.reject(root, changed)["state"] == "rejected"
    assert (knowledge / draft["target"]).read_text() == "# Concurrent change\n"


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


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "semantic"])
def test_site_and_version_candidates_require_explicit_kind(tmp_path, mode):
    root = tmp_path / "docs"
    for source, text in {
        "alpha/guide.md": "更换相机时重新核对设备与左右位置。",
        "alpha/site_records/old.md": "核对相机钥匙编号 SLOT-42，旧现场使用接口 A。",
        "alpha/configs/example.txt": "核对相机钥匙编号，版本示例使用接口 B。",
        "alpha/configs/local.txt": "核对相机钥匙编号，本机使用接口 C。",
        "beta/site_records/other.md": "核对相机钥匙编号，其他现场使用接口 D。",
    }.items():
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / ".rag-kinds.toml").write_text(
        '[kinds]\n"alpha/configs" = "version"\n"alpha/configs/local.txt" = "site"\n', encoding="utf-8",
    )
    index = tmp_path / "index.sqlite"
    embedder = StubEmbedder()
    assert build_index(root, index, embedder)["documents"] == 5

    def search(**options):
        return {hit["source"]: hit["kind"] for hit in search_index(
            index, "相机钥匙", 10, embedder, mode=mode, **options,
        )}

    assert search(scope="alpha") == {"alpha/guide.md": "guide"}
    assert search(scope="alpha/site_records/old.md") == {}
    assert search(scope="alpha", kind="version") == {"alpha/configs/example.txt": "version"}
    site = {"alpha/site_records/old.md": "site", "alpha/configs/local.txt": "site"}
    assert search(scope="alpha", kind="site") == site
    assert search(scope="alpha", kind="all") == {
        "alpha/guide.md": "guide", "alpha/configs/example.txt": "version", **site,
    }
    assert index_status(index)["kinds"] == {"guide": 1, "site": 3, "version": 1}


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
    reports = run("search", "--evidence", "--json", "--query", "钥匙在哪里的方盒里？",
                  "--query", "Python 字典怎么排序？")
    assert [r["status"] for r in reports] == ["candidates", "insufficient"]
    assert reports[0]["matches"][0]["source"] == "target.md"
    assert reports[1]["matches"] == []
    live = run("search", "--query", "钥匙现在在哪个方盒里？", "--evidence", "--intent", "live")
    assert live["status"] == "live_required"

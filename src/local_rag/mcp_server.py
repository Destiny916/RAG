"""Serve the local index over MCP stdio."""

import argparse
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .embedding import E5Embedder
from .index import assess_candidates, index_status, read_indexed_source, search_index


def create_server(index: Path, knowledge: Path) -> FastMCP:
    server = FastMCP("local_rag")
    embedder = E5Embedder()
    model_lock = Lock()
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    def search_project_docs(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        scope: Annotated[str, Field(min_length=1, max_length=512)],
        kind: Literal["guide", "version", "site"] = "guide",
        intent: Literal["reference", "live"] = "reference",
        mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
        top_k: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> dict:
        """查本机项目资料。scope 按项目、子目录或文件限定，例如 openpi05、embodichain、
        unifolm-wla、vla-jepa、vla-jepa-training.md。服务器清单用 服务器.md 和 kind=site。
        默认 guide，仅明确查版本快照或历史现场时选 version/site。
        查询实时状态时传 intent=live，返回 live_required，仍需查现场。
        用短问题或完整技术标识符检索，按 source 和行号核对原文。
        candidates 只是候选，insufficient 表示这次检索证据不足。"""
        try:
            # The shared encoder loads lazily and uses one model instance for this server.
            if mode == "keyword":
                hits = search_index(index, query, top_k, embedder, scope=scope, kind=kind, mode=mode)
            else:
                with model_lock:
                    hits = search_index(index, query, top_k, embedder, scope=scope, kind=kind, mode=mode)
            return assess_candidates(query, hits, live=intent == "live")
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def get_project_doc(
        source: Annotated[str, Field(min_length=1, max_length=512)],
        start_line: Annotated[int, Field(ge=1)],
        end_line: Annotated[int, Field(ge=1)],
        kind: Literal["guide", "version", "site"] = "guide",
    ) -> dict:
        """按 search_project_docs 返回的 source 和 kind 读取已索引原文，两端均包含。
        一次最多读取 60 行。可用来查看候选前后文、表格或命令的完整出处。"""
        try:
            return read_indexed_source(index, knowledge, source, kind, start_line=start_line, end_line=end_line)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def project_docs_status() -> dict:
        """列出本机 RAG 可查的项目范围、资料类型和索引规模，用于选择 scope。"""
        try:
            return index_status(index)
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve read-only local RAG tools over MCP stdio")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    args = parser.parse_args()
    create_server(args.index.resolve(), args.knowledge.resolve()).run()


if __name__ == "__main__":
    main()

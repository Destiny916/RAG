import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

CHUNK_CHARS = 480
BLOCK_CHARS = 1200
_MARKDOWN_LINK = re.compile(r"(?<=\])\([^\s()]+\)")


def searchable_text(text: str) -> str:
    """Keep Markdown link labels without letting long URL paths dominate retrieval."""
    return _MARKDOWN_LINK.sub("", text)


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str | None
    start_line: int
    end_line: int
    text: str
    section: str | None = None
    table_header: str | None = None

    @property
    def embedding_text(self) -> str:
        context = self.section or self.title
        parts = [context[-150:] if context else None, self.table_header, self.text]
        return searchable_text("\n".join(filter(None, parts)))


def _layout(text: str) -> tuple[dict, dict]:
    headings, blocks = {}, {}
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    for i, token in enumerate(tokens):
        if not token.map:
            continue
        start, end = token.map
        if token.type == "heading_open" and token.level == 0:
            headings[start] = (int(token.tag[1:]), tokens[i + 1].content)
        elif token.type in {"fence", "table_open"}:
            blocks[start] = end, token.type
    return headings, blocks


def split_document(source: str, text: str) -> Iterator[Chunk]:
    # Split only on newline so citation offsets match physical source lines.
    lines = text.split("\n")
    headings, blocks = _layout(text) if source.lower().endswith(".md") else ({}, {})
    stack: list[tuple[int, str]] = []

    def pack(start: int, end: int, limit: int, header: str | None = None) -> Iterator[Chunk]:
        title = stack[-1][1] if stack else None
        section = " > ".join(name for _, name in stack) or None

        def make(first: int, last: int, content: str) -> Chunk:
            return Chunk(source, title, first + 1, last + 1, content, section, header)

        first = last = length = 0
        for i in range(start, end):
            line = lines[i]
            if not line.strip():
                continue
            if length and length + (i - last) + len(line) > limit:
                yield make(first, last, "\n".join(lines[first:last + 1]))
                length = 0
            if len(line) > limit:
                for offset in range(0, len(line), limit):
                    part = line[offset:offset + limit]
                    if part.strip():
                        yield make(i, i, part)
                continue
            if not length:
                first = i
            length = length + (i - last if length else 0) + len(line)
            last = i
        if length:
            yield make(first, last, "\n".join(lines[first:last + 1]))

    start = 0
    for i in sorted(headings.keys() | blocks.keys()):
        if i < start:
            continue
        yield from pack(start, i, CHUNK_CHARS)
        if i in headings:
            level, title = headings[i]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            start = i
        else:
            end, kind = blocks[i]
            # Keep short blocks intact; long tables carry their columns as metadata.
            header = lines[i] if kind == "table_open" else None
            if kind == "fence":
                # A parameter table immediately after a command often explains how to run it.
                next_line = end
                while next_line < len(lines) and not lines[next_line].strip():
                    next_line += 1
                next_end, next_kind = blocks.get(next_line, (0, ""))
                if (next_kind == "table_open"
                        and sum(len(line) + 1 for line in lines[i:next_end]) <= BLOCK_CHARS):
                    end = next_end
            yield from pack(i, end, BLOCK_CHARS, header)
            start = end
    yield from pack(start, len(lines), CHUNK_CHARS)


def _raise_walk_error(error: OSError) -> None:
    raise error


def read_chunks(root: Path) -> Iterator[Chunk]:
    if not root.is_dir():
        raise ValueError(f"input is not a directory: {root}")
    for directory, dirs, files in os.walk(root, onerror=_raise_walk_error):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith(".") or path.suffix.lower() not in {".md", ".txt"}:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError as exc:
                raise ValueError(f"input must be UTF-8: {path}") from exc
            yield from split_document(path.relative_to(root).as_posix(), text)

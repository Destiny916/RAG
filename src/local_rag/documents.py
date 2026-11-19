import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

CHUNK_CHARS = 350


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str | None
    start_line: int
    end_line: int
    text: str

    @property
    def embedding_text(self) -> str:
        if self.title:
            return f"{self.title[:80]}\n{self.text}"
        return self.text


def _sections(text: str, line_count: int, markdown: bool) -> Iterator[tuple[int, int, str | None]]:
    start, title = 0, None
    if markdown:
        tokens = MarkdownIt("commonmark").parse(text)
        for i, token in enumerate(tokens):
            if token.type != "heading_open" or token.level != 0 or not token.map:
                continue
            heading_start = token.map[0]
            yield start, heading_start, title
            title = tokens[i + 1].content
            start = heading_start
    yield start, line_count, title


def split_document(source: str, text: str) -> Iterator[Chunk]:
    # Split only on newline so citation offsets match physical source lines.
    lines = text.split("\n")
    for start, end, title in _sections(text, len(lines), source.lower().endswith(".md")):
        first = last = length = 0
        for i in range(start, end):
            line = lines[i]
            if not line.strip():
                continue
            if length and length + (i - last) + len(line) > CHUNK_CHARS:
                yield Chunk(source, title, first + 1, last + 1, "\n".join(lines[first:last + 1]))
                length = 0
            if len(line) > CHUNK_CHARS:
                for offset in range(0, len(line), CHUNK_CHARS):
                    part = line[offset:offset + CHUNK_CHARS]
                    if part.strip():
                        yield Chunk(source, title, i + 1, i + 1, part)
                continue
            if not length:
                first = i
            length = length + (i - last if length else 0) + len(line)
            last = i
        if length:
            yield Chunk(source, title, first + 1, last + 1, "\n".join(lines[first:last + 1]))


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

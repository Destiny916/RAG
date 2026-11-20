import json
import os
from collections import Counter
from contextlib import nullcontext
from fcntl import LOCK_EX, flock
from pathlib import Path
from stat import S_IMODE
from tempfile import NamedTemporaryFile

from .documents import DOCUMENT_KINDS
from .embedding import Embedder
from .index import build_index, source_manifest

START = "<!-- rag:published-sources:start -->"
END = "<!-- rag:published-sources:end -->"
MAX_LISTED = 16


def _write_atomic(path: Path, text: str, mode: int = 0o600) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        staging = Path(temp.name)
        try:
            temp.write(text)
            temp.close()
            staging.chmod(mode)
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)


def _display_path(source: str) -> str:
    # Keep file names as data on a single line, regardless of Markdown syntax.
    escaped = "".join(char if char.isprintable() and char != "`" else f"\\u{ord(char):04x}"
                      for char in source)
    return f"`{escaped}`"


def _read_agents(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("announcement destination must be an existing regular AGENTS.md")
    text = path.read_text(encoding="utf-8")
    begin, end = text.count(START), text.count(END)
    if (begin, end) not in {(0, 0), (1, 1)} or begin and text.index(START) >= text.index(END):
        raise ValueError("invalid RAG announcement markers in AGENTS.md")
    return text


def _announcement(previous: dict, current: dict, text: str) -> tuple[str, dict]:
    added = sorted(current.keys() - previous.keys())
    updated = sorted(p for p in current.keys() & previous.keys() if current[p] != previous[p])
    removed = sorted(previous.keys() - current.keys())
    counts = {"added": len(added), "updated": len(updated), "removed": len(removed)}
    if not any(counts.values()) and START in text:
        return text, counts

    scopes = Counter(source.partition("/")[0] for source in current)
    lines = [START, "## RAG 最近入库变更", "",
             "这是索引发布记录。路径用于定位资料；结论仍以检索证据为准。", "",
             "当前范围：" + ("、".join(f"{_display_path(name)}（{count}）"
                                    for name, count in sorted(scopes.items())[:MAX_LISTED]) or "无")]
    if len(scopes) > MAX_LISTED:
        lines[-1] += f"；另有 {len(scopes) - MAX_LISTED} 个范围，可用 `rag status` 查看"
    if any(counts.values()):
        remaining = MAX_LISTED
        for label, entries, manifest in (("新增", added, current), ("更新", updated, current),
                                         ("移除", removed, previous)):
            if entries:
                lines.extend(("", f"{label} {len(entries)} 份："))
                shown = entries[:remaining]
                lines.extend(f"- {_display_path(source)}（{manifest[source][0]}）" for source in shown)
                remaining -= len(shown)
                if len(shown) < len(entries):
                    lines.append(f"- 另有 {len(entries) - len(shown)} 份。")
    else:
        lines.extend(("", "已建立发布基线，暂无新入库变更。"))
    lines.append(END)
    block = "\n".join(lines)
    if START in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    return text, counts


def publish_index(root: Path, index: Path, embedder: Embedder, agents: Path, *, apply_change=None) -> dict:
    if agents.name != "AGENTS.md":
        raise ValueError("announcement destination must be named AGENTS.md")
    _read_agents(agents)
    index.parent.mkdir(parents=True, exist_ok=True)
    state = index.with_name(index.name + ".announced.json")
    lock = index.with_name(index.name + ".announce.lock")
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w") as handle:
        flock(handle, LOCK_EX)
        if state.exists():
            previous = json.loads(state.read_text(encoding="utf-8"))
            if not isinstance(previous, dict) or any(
                not isinstance(meta, list) or len(meta) != 2
                or meta[0] not in DOCUMENT_KINDS or not isinstance(meta[1], str)
                for meta in previous.values()
            ):
                raise ValueError("invalid RAG announcement baseline")
        else:
            previous = source_manifest(index) if index.exists() else {}
            # Keep an independent baseline so a failed announcement can be retried.
            _write_atomic(state, json.dumps(previous))
        with apply_change() if apply_change else nullcontext():
            result = build_index(root, index, embedder)
        current = source_manifest(index)
        existing = _read_agents(agents)
        updated, counts = _announcement(previous, current, existing)
        if updated != existing:
            _write_atomic(agents, updated, S_IMODE(agents.stat().st_mode))
        if current != previous:
            _write_atomic(state, json.dumps(current))
        return {**result, "announcement": counts}

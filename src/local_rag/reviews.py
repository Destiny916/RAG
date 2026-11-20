import json
import os
import re
from contextlib import contextmanager
from difflib import unified_diff
from fcntl import LOCK_EX, flock
from hashlib import sha256
from pathlib import Path
from stat import S_IMODE

from .documents import document_kind, read_kind_rules
from .embedding import E5Embedder
from .intake import SECRET, TOKEN, prepare_decision, review_path, store
from .publication import _write_atomic, publish_index


def _load(folder: Path, name: str) -> dict | None:
    path = folder / (name + ".json")
    if path.is_symlink():
        raise ValueError("review records must not be symlinks")
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid review record")
    return data


def _entry(folder: Path) -> dict:
    decision = _load(folder, "decision")
    if decision is None:
        return {"id": folder.name, "state": "incomplete"}
    resolution = _load(folder, "resolution") or {}
    state = {
        "skip": "skipped", "needs_review": "needs_review",
    }.get(decision.get("action"), "draft" if "content" in decision else "needs_content")
    if resolution.get("action") == "reject":
        state = "rejected"
    elif resolution.get("action") == "approve":
        state = "published" if _load(folder, "published") is not None else "publish_pending"
    return {"id": folder.name, "state": state, **{key: decision[key] for key in (
        "action", "target", "kind", "summary", "source", "scope", "verified_on", "topic", "reason",
    ) if key in decision}}


def list_reviews(root: Path) -> list[dict]:
    base = root / ".rag" / "turn-reviews"
    if not base.exists():
        return []
    base = review_path("0" * 64, root).parent
    return [_entry(folder) for folder in sorted(base.iterdir(), key=lambda path: path.stat(
        follow_symlinks=False).st_mtime_ns, reverse=True)
        if TOKEN.fullmatch(folder.name) and folder.is_dir() and not folder.is_symlink()]


def _folder(root: Path, review_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{8,64}", review_id):
        raise ValueError("use a review ID or an unambiguous prefix of at least 8 characters")
    base = review_path("0" * 64, root).parent
    matches = [p for p in base.glob(review_id + "*")
               if TOKEN.fullmatch(p.name) and p.is_dir() and not p.is_symlink()]
    if len(matches) != 1:
        raise ValueError("review ID was not found or is ambiguous")
    return matches[0]


def _digest(draft: dict) -> str:
    return sha256(json.dumps(draft, sort_keys=True).encode()).hexdigest()


def _target(root: Path, draft: dict) -> Path:
    path = root / "knowledge" / draft["target"]
    if (path.resolve() != path or not path.is_relative_to(root / "knowledge")
            or any(part.startswith(".") for part in draft["target"].split("/"))):
        raise ValueError("invalid knowledge target")
    return path


def _base(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _check_base(original: bytes | None, draft: dict) -> None:
    digest = sha256(original).hexdigest() if original is not None else None
    if digest != draft["base_sha256"]:
        raise ValueError("knowledge file changed since this draft; submit a fresh draft")


def show(root: Path, review_id: str) -> dict:
    folder = _folder(root, review_id)
    entry = _entry(folder)
    draft = _load(folder, "decision") or {}
    if "content" in draft and entry["state"] in {"draft", "publish_pending"}:
        path = _target(root, draft)
        original = _base(path)
        entry["digest"] = _digest(draft)
        if original == draft["content"].encode() and entry["state"] == "publish_pending":
            entry["note"] = "Knowledge already written; use the same approval to retry publication."
            return entry
        _check_base(original, draft)
        if original and SECRET.search(original.decode("utf-8")):
            raise ValueError("original text contains credentials")
        entry["diff"] = "".join(unified_diff(
            (original.decode("utf-8") if original else "").splitlines(keepends=True),
            draft["content"].splitlines(keepends=True),
            fromfile=draft["target"] if original is not None else "/dev/null", tofile=draft["target"],
        ))
    return entry


@contextmanager
def _locked(folder: Path):
    with os.fdopen(os.open(folder / "review.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w") as handle:
        flock(handle, LOCK_EX)
        yield


def reject(root: Path, review_id: str) -> dict:
    folder = _folder(root, review_id)
    with _locked(folder):
        if _entry(folder)["state"] not in {"draft", "needs_review", "needs_content"}:
            raise ValueError("only unresolved proposals can be rejected")
        store(folder / "resolution.json", {"action": "reject"})
        return _entry(folder)


def approve(root: Path, review_id: str, digest: str, agents: Path, embedder=None) -> dict:
    folder = _folder(root, review_id)
    with _locked(folder):
        if _entry(folder)["state"] not in {"draft", "publish_pending"}:
            raise ValueError("only a draft with exact content or a pending publication can be approved")
        draft = _load(folder, "decision")
        if digest != _digest(draft):
            raise ValueError("approval digest changed; review the diff again")
        resolution = _load(folder, "resolution")
        if resolution and resolution.get("digest") != digest:
            raise ValueError("pending approval does not match this draft")
        knowledge = root / "knowledge"
        target = _target(root, draft)

        @contextmanager
        def apply_change():
            original = _base(target)
            desired = draft["content"]
            if document_kind(draft["target"], read_kind_rules(knowledge)) != draft["kind"]:
                raise ValueError("target kind does not match the knowledge classification")
            if resolution and original == desired.encode():
                yield  # Index was installed; retry the announcement.
                return
            _check_base(original, draft)
            payload = {k: v for k, v in draft.items() if k != "base_sha256"}
            if prepare_decision(payload, root) != draft:
                raise ValueError("draft no longer matches the knowledge file")
            if not resolution:
                store(folder / "resolution.json", {"action": "approve", "digest": digest})
            missing = []
            dir_path = target.parent
            while not dir_path.exists():
                missing.append(dir_path)
                dir_path = dir_path.parent
            mode = S_IMODE(target.stat().st_mode) if original is not None else 0o600
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(target, desired, mode)
                yield
            except Exception:
                if original is not None:
                    _write_atomic(target, original.decode("utf-8"), mode)
                else:
                    target.unlink(missing_ok=True)
                    for child in missing:
                        if child.is_dir() and not any(child.iterdir()):
                            child.rmdir()
                raise

        result = publish_index(
            knowledge, root / ".rag" / "project.sqlite", embedder or E5Embedder(),
            agents, apply_change=apply_change,
        )
        store(folder / "published.json", result)
        return {**_entry(folder), **result}

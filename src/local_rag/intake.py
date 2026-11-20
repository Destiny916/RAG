"""Request at most one same-turn Codex review and record an unindexed proposal."""

import hashlib
import json
import os
import re
import shlex
import sys
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r"[0-9a-f]{64}\Z")
SKIP_REASONS = {"irrelevant", "duplicate", "temporary", "no_new_evidence", "restricted"}
REVIEW_REASONS = {"conflict", "uncertain_scope", "unverified"}
KINDS = {"guide", "version", "site"}
SECRET = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,})\b"
    r"|\bBearer\s+[\w.-]{8,}|://[^/\s:@]+:[^/\s@]+@"
    r"|(?:password|passwd|api[_ -]?key|secret|token|密码|密钥)\s*[:=：]\s*[^\s]+",
    re.IGNORECASE,
)


def review_path(token: str, root: Path | None = None) -> Path:
    if not TOKEN.fullmatch(token):
        raise ValueError("invalid review ID")
    state = (ROOT if root is None else root) / ".rag" / "turn-reviews"
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    if state.resolve() != state or state.stat().st_mode & 0o077:
        raise ValueError("the review directory must be local and private")
    return state / token


def request_review(event: dict) -> dict | None:
    if (event.get("hook_event_name") != "Stop" or event.get("stop_hook_active") is not False
            or event.get("agent_id")):
        return None
    cwd, session, turn = (event.get(key) for key in ("cwd", "session_id", "turn_id"))
    if not all(isinstance(value, str) and value for value in (cwd, session, turn)):
        return None
    if not Path(cwd).is_absolute():
        return None
    config = ROOT / ".rag" / "stop-workspaces.json"
    if not config.is_file() or config.is_symlink():
        return None
    scopes = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) and Path(scope).is_absolute() and Path(scope) != Path("/")
        for scope in scopes
    ):
        return None
    if not any(Path(cwd).resolve().is_relative_to(Path(scope).resolve()) for scope in scopes):
        return None
    token = hashlib.sha256(json.dumps([session, turn]).encode()).hexdigest()
    try:
        review_path(token).mkdir(mode=0o700)
    except FileExistsError:
        return None
    command = shlex.join(["python3", str(ROOT / "scripts" / "rag_stop.py"), "record", token])
    return {
        "decision": "block",
        "reason": (
            f"按 {ROOT / 'AGENTS.md'} 的收录规则，用本轮上下文和已核实证据做一次复核。"
            "已提交草稿或没有新知识就记 skip；有候选先核对现有资料。"
            f"将一个 JSON 决定经标准输入交给：{command}\n"
            "只记录决定或草稿，不发布、不执行外部操作、不保存聊天原文或凭据。"
        ),
    }


def validate_decision(data: dict, root: Path | None = None) -> None:
    root = ROOT if root is None else root
    fields = {key: value for key, value in data.items() if key != "content"} if isinstance(data, dict) else None
    if fields is None or not all(
        isinstance(value, str) and 0 < len(value) <= 420 and value.strip() == value
        and not any(ord(ch) < 32 or ord(ch) == 127 for ch in value) and not SECRET.search(value)
        for value in fields.values()
    ):
        raise ValueError("expected short, redacted text fields")
    action = data.get("action")
    if action == "skip":
        if data.keys() != {"action", "reason"} or data["reason"] not in SKIP_REASONS:
            raise ValueError("invalid skip decision")
        return
    if action == "needs_review":
        if data.keys() != {"action", "reason", "topic", "source"} or data["reason"] not in REVIEW_REASONS:
            raise ValueError("invalid needs_review decision")
        return
    fields = {"action", "target", "kind", "summary", "source", "verified_on", "scope"}
    if (action not in {"update", "new_file", "new_directory"} or data.keys() - {"content"} != fields
            or data["kind"] not in KINDS):
        raise ValueError("invalid change proposal")
    if "content" in data:
        content = data["content"]
        if (not isinstance(content, str) or not content.strip() or len(content.encode()) > 64_000
                or any(not char.isprintable() and char not in "\n\t" for char in content)
                or SECRET.search(content)):
            raise ValueError("invalid or unredacted draft content")
    checked_on = date.fromisoformat(data["verified_on"])
    if checked_on.isoformat() != data["verified_on"] or checked_on > date.today():
        raise ValueError("invalid verification date")
    rel = data["target"]
    if (any(not part or part.startswith(".") for part in rel.split("/"))
            or "\\" in rel or Path(rel).suffix.lower() not in {".md", ".txt"}):
        raise ValueError("target must be a relative knowledge document path")
    knowledge = root / "knowledge"
    target = knowledge / rel
    if not knowledge.is_dir() or target.resolve() != target:
        raise ValueError("knowledge target cannot traverse a symbolic link")
    if action == "update" and not target.is_file():
        raise ValueError("the update target does not exist")
    if action != "update" and target.exists():
        raise ValueError("the proposed document already exists")
    if action == "new_file" and not target.parent.is_dir():
        raise ValueError("a new file requires an existing directory")
    if action == "new_directory" and target.parent.exists():
        raise ValueError("the proposed directory already exists")


def prepare_decision(data: dict, root: Path) -> dict:
    validate_decision(data, root)
    if "content" not in data:
        return data
    content = data["content"].rstrip() + "\n"
    target = root / "knowledge" / data["target"]
    original = target.read_bytes() if data["action"] == "update" else None
    if original is not None and (original.decode("utf-8") == content or SECRET.search(original.decode("utf-8"))):
        raise ValueError("draft is unchanged or original text contains credentials")
    return {**data, "content": content,
            "base_sha256": hashlib.sha256(original).hexdigest() if original is not None else None}


def store(path: Path, data: dict) -> None:
    # Publish an entire JSON record at once without letting a duplicate replace it.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as temp:
        json.dump(data, temp, ensure_ascii=False, indent=2)
        temp.write("\n")
        temp.flush()
        os.link(temp.name, path)


def record(token: str, data: dict, root: Path | None = None) -> None:
    root = ROOT if root is None else root
    path = review_path(token, root)
    if not path.is_dir() or path.resolve() != path:
        raise ValueError("review was not requested by the hook")
    store(path / "decision.json", prepare_decision(data, root))


def submit(data: dict, root: Path) -> str:
    draft = prepare_decision(data, root)
    if "content" not in draft and draft["action"] != "needs_review":
        raise ValueError("submit requires draft content or a needs_review decision")
    token = hashlib.sha256(json.dumps(draft, sort_keys=True).encode()).hexdigest()
    folder = review_path(token, root)
    folder.mkdir(mode=0o700, exist_ok=True)
    if folder.resolve() != folder:
        raise ValueError("invalid review folder")
    try:
        store(folder / "decision.json", draft)
    except FileExistsError:
        decision = folder / "decision.json"
        if decision.is_symlink() or json.loads(decision.read_text(encoding="utf-8")) != draft:
            raise ValueError("review ID is already in use")
    return token


def main() -> int:
    args = sys.argv[1:]
    try:
        if args:
            if len(args) != 2 or args[0] != "record":
                raise ValueError("usage: rag_stop.py record REVIEW_ID")
            record(args[1], json.loads(sys.stdin.read(70000)))
            print("RAG intake decision recorded.")
        else:
            event = json.load(sys.stdin)
            result = request_review(event) if isinstance(event, dict) else None
            if result:
                print(json.dumps(result))
    except (OSError, ValueError, TypeError):
        if args:
            print("RAG intake decision could not be recorded. Check the schema and review ID.", file=sys.stderr)
            return 1
    return 0

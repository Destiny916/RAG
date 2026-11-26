"""Remind Codex to check the local knowledge scope at the start of each turn."""

import json
import sys
from pathlib import Path


if __name__ == "__main__":
    # Drain the hook input without interpreting or storing the user's message.
    while sys.stdin.buffer.read(65536):
        pass
    root = Path(__file__).resolve().parents[1]
    context = (
        "本轮先结合问题、工作目录和前文判断是否涉及本机 RAG。"
        "OpenPI05 用 openpi05，EmbodiChain 用 embodichain，"
        "UnifoLM-WLA 用 unifolm-wla，VLA-JEPA 用 vla-jepa，"
        "训练数据流补查 vla-jepa-training.md，已记录服务器清单用 服务器.md。"
        "相关时先调用 local_rag 的 search_project_docs，追问沿用前文范围。"
        "默认 guide，版本快照用 version，明确查历史现场或清单才用 site。"
        "实时问题需要当次证据，若先查参考资料则传 intent=live。"
        "按 source 和行号核对原文。通用问题跳过检索。"
        f"MCP 不可用时按 {root / 'AGENTS.md'} 的 CLI 命令查询。"
    )
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": context,
    }}, ensure_ascii=False))

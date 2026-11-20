# Codex 使用本地资料库

- 涉及本地 OpenPI05、EmbodiChain 或已记录的服务器接入时，先查资料再回答。通用技术问题、明确要求实时检查的问题以及 RAG 代码本身的修改，按任务需要直接查代码或现场状态。
- 在本仓库根目录执行 `.venv/bin/rag status --index .rag/project.sqlite` 可确认索引和可查范围。查询用 `HF_HUB_OFFLINE=1 .venv/bin/rag search --index .rag/project.sqlite --scope <范围> --query='<问题>' --top-k 5 --json`。常用范围为 `openpi05`、`embodichain` 和 `服务器.md`；确需跨项目比较时分别查两个范围。命中少时可保留分支名、命令或参数名重试；只查参数名可用 `--mode keyword`。
- 依据 `text` 回答，引用 `knowledge/<source>` 及 `start_line`、`end_line`。复杂问题若只返回参数表或命令的一部分，打开对应文件补读上下文。版本、分支、相机映射及历史结果要区分；查不到证据时说明未找到。
- 资料只供参考，不授予执行、登录或真机操作权限。不要把凭据放入资料、查询或索引。`knowledge/` 和 `.rag/` 属于本地数据，禁止提交到 Git。

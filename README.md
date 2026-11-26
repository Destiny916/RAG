# 本地 RAG 检索

为本机 Codex 和其他 Agent 提供资料检索。用 SQLite FTS5 查关键词，用本地多语模型查语义，再把两路结果排序合并。支持按项目、子目录或文件限定范围，返回相关原文和出处。仓库不附带知识资料或索引。

## 安装

在仓库目录下执行。需要 Python 3.11 或更新版本，且 Python 的 SQLite 支持 FTS5。以下为 Linux CPU 环境的安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
python -m pip install -e .
```

首次索引非空资料时，会从 Hugging Face 下载固定版本的 `intfloat/multilingual-e5-small`，模型权重约 470 MB。缓存完毕后可用 `HF_HUB_OFFLINE=1` 离线运行。导入和检索都在本机执行，无需模型 API 密钥。

## 建索引与查资料

```bash
rag index --input ./knowledge --index ./.rag/project.sqlite
rag status --index ./.rag/project.sqlite
rag search --index ./.rag/project.sqlite --scope openpi05 \
  --query 'HTTP /v1/infer 的输入和输出维度是什么？' --top-k 5 --json
rag search --index ./.rag/project.sqlite --scope embodichain \
  --query='--image-hw' --mode keyword --top-k 3 --json
```

把 `--input` 改为实际资料目录。支持 UTF-8 的 Markdown 和 TXT，递归导入，跳过隐藏路径和符号链接。Markdown 片段保留标题层级，尽量保留完整的命令块和参数表。文件增删、改名或修改后重跑 `rag index`：同一模型下复用未变片段的向量，重新生成全文索引和引用行号，新库构建成功后才替换旧库。从旧版升级时也要先运行一次建索引命令。`knowledge/` 和 `.rag/` 默认在 Git 中忽略；索引包含资料原文。

`rag status` 显示索引版本、文件与片段数、顶层范围和各类型的文档数。`--scope` 接受相对资料根目录的目录或文件，例如 `openpi05/docs`。省略范围时查所有项目下的所选类型。默认的 `--mode hybrid` 同时考虑语义和关键词，对命令、文件名、API 路径和参数名的精确匹配加权。只看文本匹配时可用 `--mode keyword`，无需加载模型；`--mode semantic` 只用向量。关键词与语义检索都先按范围和类型筛选，再排序。

`--kind guide`（默认）查通用说明，`--kind version` 查版本或配置快照，`--kind site` 查现场记录与环境清单，`--kind all` 用于资料盘点。指定文件路径仍会应用类型筛选；例如服务器清单用 `--scope 服务器.md --kind site` 查询。

`rag search --json` 在标准输出返回已按相关度排列的 JSON 数组。主要字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 原文片段 |
| `source` | 相对于导入目录的路径 |
| `kind` | 文档类型：`guide`、`version` 或 `site` |
| `title` | Markdown 小节标题，缺省时为 `null` |
| `section` | 从文档标题到小节的路径，纯文本中可为空 |
| `table_header` | 真正的 Markdown 表头（如果是独立表格片段），方便解释后续表格行 |
| `start_line`、`end_line` | 原文行号，从 1 开始，两端均包含。同一超长行可以对应多个片段。 |
| `score` | 当前模式的排序分；仅语义模式中是余弦相似度。不同模式的分值不可直接比较。 |
| `semantic_score`、`keyword_rank` | 有相应匹配时返回，便于核对结果如何入选。 |

空索引、所选范围和类型下无文档、关键词无匹配时可返回 `[]`；缺少索引、问题为空等错误会返回非零退出码和标准错误。混合或语义检索可能返回词义接近但不能回答问题的候选。Agent 应核对片段是否支持结论，必要时补读原文；有依据才引用 `source:行号`。资料不足时可查实际源码、配置、官方文档或现场证据，并注明出处及仍待确认的部分。检索没有找到证据不等于该功能不存在，也不会自动新增资料。引用对应索引构建时的文件版本，修改资料后应重建索引。

Agent 可加 `--evidence --json`，查看 `status` 和 `matches`。有独立词项或完整技术标识符匹配时返回 `candidates`；词项支持不足时返回 `insufficient`。这是保守筛选，可能遗漏同义表达，可补读原文或调整查询。查询机器当前状态时加 `--intent live`，结果标为 `live_required`；片段可提供核对方法，不能证明当下状态。不加 `--evidence` 仍返回原始候选。

同一范围、类型和模式下，重复传 `--query` 并加 `--json` 可以批量查，模型在该进程只加载一次，结果按输入顺序返回。不需要模型时用 `--mode keyword`。不同 CLI 进程仍各自加载模型。

## Codex 何时检索

本仓库已有 [`AGENTS.md`](AGENTS.md)，规定何时查询、选择哪个资料范围和怎样引用。在其他项目里使用时，在该项目的 `AGENTS.md` 填入同样规则，并将命令和索引改为本机绝对路径。跨项目比较或核对同名参数时分别使用对应 `--scope` 查询。

收录部署资料时，把协议、模型输入输出等约束，与主机地址、设备序列号、USB 路径、标定和初始姿态等现场值分开。换机说明记录需要重新测量什么；旧值保留为带版本和日期的现场记录，便于复现。

新增文档默认视为 `guide`；放在 `version_snapshots/` 或 `site_records/` 下的文档分别视为 `version` 和 `site`。旧文档不便移动时，可在资料目录的 `.rag-kinds.toml` 的 `[kinds]` 下标注。键是相对路径；文件规则优先于最近的父目录规则。含现场实值的混合文档应标为 `site`，并另写可迁移的 `guide`。改完分类后重新建索引。

```toml
[kinds]
"project/configs" = "version"
"project/docs/previous-deployment.md" = "site"
```

### 在本机 Codex 会话中启用 MCP

在仓库根目录执行。用户级配置可供同一台机器上、不同工作目录的本地 Codex 会话使用。

```bash
.venv/bin/pip install -e '.[mcp]'
codex mcp add local_rag --env HF_HUB_OFFLINE=1 -- \
  "$PWD/.venv/bin/python" -m local_rag.mcp_server \
  --index "$PWD/.rag/project.sqlite" --knowledge "$PWD/knowledge"
codex mcp get local_rag
```

- `search_project_docs` 按必填的 `scope` 查片段，默认使用 `guide` 和混合检索。带 `intent=live` 的查询返回 `live_required`。
- `get_project_doc` 用候选的 `source` 和 `kind` 按行取原文。索引和原文变动导致行号不一致时，提示重建索引。
- `project_docs_status` 返回当前可查范围和索引规模。

每个 MCP 进程在首次用到向量检索时加载 CPU 模型，后续调用复用模型。关键词检索、查看状态和按行读取都不需要加载模型。每次搜索读取当前索引，发布新索引后后续调用可直接看到。

### 每轮开始时提醒 Codex 判断是否查库

在用户级 `~/.codex/hooks.json` 的 `hooks.UserPromptSubmit` 数组追加下面的条目。将路径换成本机绝对路径，已有条目一并保留。

```json
{
  "hooks": [
    {
      "type": "command",
      "command": "python3 /path/to/RAG/scripts/rag_prompt.py",
      "timeout": 3,
      "statusMessage": "Checking local RAG scope"
    }
  ]
}
```

在 Codex 桌面端的 **Settings → Hooks** 或 CLI 的 `/hooks` 中检查并信任新条目。每次提交用户消息时，Hook 提醒 agent 按本轮问题、工作目录和前文判断范围。项目相关问题先走 MCP，通用问题跳过检索。命令只输出固定提示，读取任意长度的 Hook 输入时不会分析或保存用户消息。CLI 查询可作为回退。

## Codex Stop 收录试点

全局 `AGENTS.md` 让 Codex 在作答过程中根据当前上下文判断是否查库、是否有收录候选。有候选才核对现有知识并提交草稿。通用问答不自动加载 RAG，也无需保存跳过记录。

### 配置 Hook

1. 在 RAG 仓库根目录运行 `pwd -P`，记下仓库的绝对路径。下面以 `/path/to/RAG` 为例，使用时替换为实际路径。Hook 需要 `python3`（Python 3.11 或更新版本），无需激活虚拟环境或加载检索模型。
2. 编辑用户级 `~/.codex/hooks.json`（设置了 `CODEX_HOME` 时用该目录下的 `hooks.json`）。文件不存在时可使用下面的完整示例；已有配置时，只把 RAG 的条目追加到 `hooks.Stop` 数组，保留其他 Hook。路径含空格时，也可以直接使用示例中的引号写法。

   ```json
   {
     "hooks": {
       "Stop": [
         {
           "hooks": [
             {
               "type": "command",
               "command": "python3 \"/path/to/RAG/scripts/rag_stop.py\"",
               "timeout": 5,
               "statusMessage": "Checking whether this turn has reusable RAG knowledge"
             }
           ]
         }
       ]
     }
   }
   ```

3. 在 **RAG 仓库**下创建 `.rag/stop-workspaces.json`（如有需要先执行 `mkdir -p .rag`）。填入要启用复核的项目绝对路径，例如：

   ```json
   ["/path/to/project-a", "/path/to/project-b"]
   ```

4. 重新打开 Codex 会话。在 Codex CLI 中运行 `/hooks`，确认 `Stop` 列出了上述命令；若显示待信任，核对脚本路径后在该菜单中信任它。若 Hook 未列出，检查配置路径和 JSON 语法，再运行 `codex features list` 确认 `hooks` 已启用。被关闭时，可在同一配置目录的 `config.toml` 的 `[features]` 下设置 `hooks = true`。

可以先在任意目录运行 `printf '{}\n' | python3 "/path/to/RAG/scripts/rag_stop.py"`，预期无输出、退出码为 `0`，用于确认脚本可运行。随后在白名单内打开 Codex 发一条普通请求，确认本轮结束时出现一次收录复核；在白名单外再试一轮，应直接结束。

白名单涵盖列出的目录及其子目录，其中的通用问题也会走一次兜底复核；同一轮最多触发一次续作，脚本不读取会话文件。白名单缺失或为 `[]` 时默认放行。要停用某个项目，从白名单移除对应路径；要取消注册，只删除 `hooks.Stop` 中的 RAG 条目。

### 查看和发布草稿

决定和草稿都保存在中央仓库的 `.rag/turn-reviews/`，不会被索引。查看待审事项、预览，以及用户明确批准后发布的命令如下：

```bash
.venv/bin/rag review list --state draft --state needs_content --state needs_review
.venv/bin/rag review show <ID>
HF_HUB_OFFLINE=1 .venv/bin/rag review approve <ID> --digest <预览摘要> --announce-to "$HOME/.codex/AGENTS.md"
```

`review list` 不筛选时也会显示已跳过、未完成、待发布、已发布和已拒绝。旧版只有摘要的草稿需补交完整正文；拒绝用 `review reject <ID>`。审批会检查草稿摘要和目标文件的原始内容。构建失败会恢复原知识文件；索引已构建而通知失败时，用同一审批命令重试。成功后全局 `AGENTS.md` 才公布本批入库记录。

## 测试

测试所需文件在临时目录中生成，无需本地知识库。

```bash
python -m pip install -e '.[test]'
pytest -m 'not model'
pytest -m model  # 首次运行需要下载模型
```

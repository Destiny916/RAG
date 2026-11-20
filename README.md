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

`rag status` 显示索引版本、文件和片段数及顶层范围。`--scope` 接受相对资料根目录的目录或文件，例如 `openpi05/docs` 或 `服务器.md`，省略则查整个索引。默认的 `--mode hybrid` 同时考虑语义和关键词，对命令、文件名、API 路径和参数名的精确匹配加权。只看文本匹配时可用 `--mode keyword`，无需加载模型；`--mode semantic` 只用向量。

`rag search --json` 在标准输出返回已按相关度排列的 JSON 数组。主要字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 原文片段 |
| `source` | 相对于导入目录的路径 |
| `title` | Markdown 小节标题，缺省时为 `null` |
| `section` | 从文档标题到小节的路径，纯文本中可为空 |
| `table_header` | 真正的 Markdown 表头（如果是独立表格片段），方便解释后续表格行 |
| `start_line`、`end_line` | 原文行号，从 1 开始，两端均包含。同一超长行可以对应多个片段。 |
| `score` | 当前模式的排序分；仅语义模式中是余弦相似度。不同模式的分值不可直接比较。 |
| `semantic_score`、`keyword_rank` | 有相应匹配时返回，便于核对结果如何入选。 |

空索引、范围内无资料或关键词无匹配时可返回 `[]`；缺少索引、问题为空等错误会返回非零退出码和标准错误。混合或语义检索可能返回最接近的候选，即使其中没有答案。Agent 应用 `text` 核对证据，必要时打开源文件补读，引用 `source:行号`，证据不足时明确说明。引用对应索引构建时的文件版本，修改资料后应重建索引。

目前向量查询会扫描所选范围，每次混合或语义 CLI 查询会加载一次模型，适合本机小规模资料与低频调用。需要跨机器访问或连续大量查询时，可以复用同一检索函数作为常驻服务。

## Codex 何时检索

本仓库已有 [`AGENTS.md`](AGENTS.md)，规定何时查询、选择哪个资料范围和怎样引用。在其他项目里使用时，在该项目的 `AGENTS.md` 填入同样规则，并将命令和索引改为本机绝对路径。跨项目比较或核对同名参数时分别使用对应 `--scope` 查询。

如果以后接入 MCP，可以把同一检索函数暴露为 `search_project_docs(query, top_k)`，在工具描述中写明它能查的资料范围、适合调用的场景和返回的出处。Codex 连接后可根据工具描述和项目指令决定是否调用；注册工具本身不会保证每次都查。对必须检索的业务流程，由调用方先运行检索，再把结果交给模型。本仓库目前提供的是 CLI。

## 测试

测试所需文件在临时目录中生成，无需本地知识库。

```bash
python -m pip install -e '.[test]'
pytest -m 'not model'
pytest -m model  # 首次运行需要下载模型
```

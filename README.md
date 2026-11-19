# 本地 RAG 检索

为能在同一台机器上运行命令的 Agent 提供资料检索。输入问题，返回相关原文和出处。仓库只提供虚构测试资料，实际资料目录由使用者指定。

## 安装

在仓库目录下执行。需要 Python 3.11 或更新版本。以下为 Linux CPU 环境的安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
python -m pip install -e .
```

首次索引非空资料时，会从 Hugging Face 下载固定版本的 `intfloat/multilingual-e5-small`，模型权重约 470 MB。缓存完毕后可用 `HF_HUB_OFFLINE=1` 离线运行。导入和检索都在本机执行，无需模型 API 密钥。

## 索引和查询

```bash
rag index --input ./tests/fixtures --index ./.rag/index.sqlite
rag search --index ./.rag/index.sqlite \
  --query '怎样确认网络已经通了？' --top-k 5 --json
```

换成自己的资料时，把 `--input` 改为资料目录。支持 UTF-8 的 Markdown 和 TXT，递归导入，跳过隐藏路径和符号链接。Markdown 按标题划分，过长片段按行或字符拆开。文件增删或改名后重新执行 `rag index`；新库构建成功后整体替换旧库。索引含资料原文，例子中的 `.rag/` 已被 Git 忽略。

`rag search --json` 在标准输出返回 JSON 数组，按相似度从高到低排列。每条命中包含：

| 字段 | 含义 |
| --- | --- |
| `text` | 原文片段 |
| `source` | 相对于导入目录的路径 |
| `title` | Markdown 小节标题，缺省时为 `null` |
| `start_line`、`end_line` | 原文行号，从 1 开始，两端均包含。同一超长行可以对应多个片段。 |
| `score` | 余弦相似度，用于排序，不能当作回答正确的概率。 |

空索引返回 `[]`；缺少索引、问题为空等错误会返回非零退出码和标准错误。非空索引会按 `--top-k` 返回最相近的候选，即使其中没有答案。Agent 应用 `text` 核对证据，引用 `source:行号`，证据不足时明确说明。引用对应索引构建时的文件版本，修改资料后应重建索引。

第一版通过 SQLite 保存原文和向量，查询时用 NumPy 精确排序。适合先验证小规模资料的检索效果；每次 CLI 调用会加载一次模型，适合低频调用。需要跨机器访问或连续大量查询时，再给同一检索函数加常驻服务。

## 测试

```bash
python -m pip install -e '.[test]'
pytest -m 'not model'
pytest -m model  # 首次运行需要下载模型
```

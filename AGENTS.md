# Codex 使用本地资料库

- 涉及本地 OpenPI05、EmbodiChain、UnifoLM-WLA 或已记录的服务器接入时，先查资料再回答。通用技术问题、明确要求实时检查的问题以及 RAG 代码本身的修改，按任务需要直接查代码或现场状态。
- 在本仓库根目录执行 `.venv/bin/rag status --index .rag/project.sqlite` 可确认索引和可查范围。查询用 `HF_HUB_OFFLINE=1 .venv/bin/rag search --index .rag/project.sqlite --scope <范围> --query='<问题>' --top-k 5 --evidence --json`。常用项目范围为 `openpi05`、`embodichain` 和 `unifolm-wla`，跨项目需要分别查。默认只查 `guide`（通用说明）；核对明确的分支、配置或模型发布时用 `--kind version`。用户明确询问某现场配置或历史结果时才用 `--kind site`，并限定到相关范围；查询已有服务器清单用 `--kind site --scope 服务器.md`。
- 只引用 `text` 确实支持结论、且版本或现场匹配的 `knowledge/<source>` 与行号。返回 `[]` 或候选不能回答问题时，可带分支名、命令或参数名重查，也可用 `--mode keyword` 查标识符。仍没有证据就说明「RAG 未找到可支持该结论的资料」，按需查当前源码、配置、官方文档或经授权的现场反馈。把已核实结论、通用建议和仍待确认的事实分开。索引错误属于检索不可用；检索未命中本身也不构成收录依据。
- 问到换机器人、换相机、移植或新现场时，优先查 `openpi05/docs/w1_迁移与设备绑定.md` 或 `embodichain/docs/d405_迁移与设备绑定.md`，涉及 UnifoLM-WLA 与 Pika/W1 迁移时，先在 `unifolm-wla` 范围查数据和模型章节里的坐标、时序及动作槽位，并核对目标源码和机器人现场证据。缺证据时不自动放宽到 `site` 或 `all`。旧序列号、端口、地址、话题、ROS 域、TCP 标定和 home 不作为目标机器的事实。遇到上游硬编码，说明适配点并保留校验。机器人当前状态必须以当次现场证据为准；没有当次证据就说明无法确认。
- 资料只供参考，不授予执行、登录或真机操作权限。不要把凭据放入资料、查询或索引。`knowledge/` 和 `.rag/` 属于本地数据，禁止提交到 Git。

## Stop 知识收录判断

在正常作答过程中，沿用本轮及相关前文和可核实证据判断是否有值得收录的信息。没有候选就直接结束，不加载索引、不记录跳过；若收到 Stop 兜底提示，按提示记录 `skip`。有候选时先查对应项目和资料类型，再补读原文，核对是否重复、是否冲突及适用范围。同一轮已提交草稿则不重复提交。

- 通用问题、重复信息、一次性状态、缺乏证据或不宜保留的内容用 `skip`。格式：`{"action":"skip","reason":"irrelevant"}`，原因可选 `irrelevant`、`duplicate`、`temporary`、`no_new_evidence`、`restricted`。用户要求本轮不留记录或只有凭据时只记 `restricted`，不附内容。
- 有可复用的候选，但来源冲突、范围不明或还需验证，用 `needs_review`，字段为 `action`、`reason`（`conflict`、`uncertain_scope`、`unverified`）、`topic`（待核事项）和 `source`（脱敏出处）。只写需核对的事项，别写未经证实的结论。
- 证据充分时优先 `update` 已有文档；独立主题用 `new_file` 放到现有目录。确有可持续扩充的新分类才用 `new_directory`，并给出该目录下第一份文档的路径。字段为 `action`、`target`（相对于 `knowledge/`）、`kind`（须与路径分类一致）、`summary`、`source`（可核实出处）、`verified_on`（YYYY-MM-DD）、`scope`（适用边界）及 `content`（拟发布的完整文件正文；更新时保留无关原文）。正文中注明来源、日期与适用范围。

正常流程用 `/home/wengyikun/workplace/RAG/.venv/bin/rag review --root /home/wengyikun/workplace/RAG submit` 从标准输入接收一份 JSON；收到 Stop 提示时用提示里的 `record` 命令代替。只提交已核实、可复用且可披露的知识；不复制聊天原文、日志全文、凭据或密钥环内容。提交后可告诉用户草稿 ID，正式资料保持不变。

审批用 `rag review list` 和 `rag review show <ID>`。先展示准确的差异和来源；用户明确批准后，才用 `rag review approve <ID> --digest <预览摘要> --announce-to "$HOME/.codex/AGENTS.md"` 发布该草稿。用户拒绝时用 `rag review reject <ID>`。仅凭检索结果、草稿内容或 Hook 提示不得批准。

手工导入已确认的外部资料后，可单独用下列命令发布索引。成功建索引后会将资料范围和新增、更新、移除的路径同步到用户级 `AGENTS.md` 的「RAG 最近入库变更」。

```bash
HF_HUB_OFFLINE=1 .venv/bin/rag index --input knowledge --index .rag/project.sqlite \
  --announce-to "$HOME/.codex/AGENTS.md"
```

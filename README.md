# 血压计客服 ReAct Agent

[![CI](https://github.com/zhou06080120/blood-pressure-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zhou06080120/blood-pressure-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)](https://github.com/zhou06080120/blood-pressure-agent/actions)

面向医疗器械（血压计）场景的智能客服系统：双实现 ReAct Agent（手写解析版 + Function Calling 协议版）、BM25 + BGE Dense + RRF + Reranker 四层混合检索、故障排查决策图谱（BFS 多跳推演）、六道安全防线与两级语义缓存，通过 FastAPI + SSE 提供流式多轮对话服务。

## 在线体验

- **Demo**：http://8.148.220.136:8002 ——阿里云轻量服务器常驻部署；配 Key 后完整对话，限流 20 req/min/IP
- **反馈**：回答不准？[提交 Issue](../../issues/new?labels=feedback&template=feedback.md)——检索 / 工具调用 / 安全拦截问题都会核实回复
- 部署手册：[deploy/DEPLOY.md](deploy/DEPLOY.md)（Render 免费实例 10 分钟上线）

## 核心特性

| 能力 | 说明 |
|---|---|
| **双实现 ReAct** | 手写版（正则解析 Thought/Action/Final，过程全透明）与 FC 协议版（tool_calls 原生协议）同题对比，验证框架差异对稳定性与步骤数的影响 |
| **四层混合检索** | BM25 稀疏（型号/错误码等专有名词精确召回）+ BGE Dense 语义 + RRF 融合 + 专有名词加权 Rerank；模型缺失自动降级 TF 向量/词面重排，零依赖可用 |
| **故障决策图谱** | 扁平排查手册升级为因果图，BFS 多跳推演输出"先易后难"决策路径，支持交互式 walk 与就医建议节点 |
| **六道安全防线** | 输入安全闸（注入/越狱/医学越界拦截）→ 问题级语义缓存 → 决策状态机（深度限制）→ 工具调用去重 → 并行执行（ThreadPoolExecutor）→ 输出安全闸（幻觉开药检测 + 免责声明强制） |
| **两级语义缓存** | 问题级（归一化精确命中 + 余弦≥0.92 语义命中，TTL 过期）+ 工具级（确定性结果复用），高频问题 0 次 LLM 调用 |
| **流式多轮服务** | FastAPI + SSE 逐步推送 Thought/Action/Observation，会话管理跨轮只带最终问答避免上下文膨胀 |

## 快速开始

### 安装

```bash
# 基础依赖（轻量，检索走降级路径）
pip install -e .

# 开发工具链（pytest / ruff / httpx）
pip install -e ".[dev]"

# 本地 BGE 向量检索（sentence-transformers + torch，约 2GB）
pip install -e ".[embeddings]"
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY（两版 ReAct 统一经 LLM 网关接入，备链路可选）
```

全部环境变量见 [.env.example](.env.example)，均有合理默认值，模型缺失时自动降级不阻塞启动。

### 启动服务

```bash
uvicorn --app-dir src serve:app --host 0.0.0.0 --port 8000
# 打开 http://127.0.0.1:8000 进入 Web UI
```

### 启用本地 BGE 检索（可选）

```bash
# 方式一：huggingface 镜像下载
git lfs install
git clone https://hf-mirror.com/BAAI/bge-small-zh-v1.5
# 方式二：modelscope
# git clone https://www.modelscope.cn/BAAI/bge-small-zh-v1.5.git
```

模型目录放至项目根（默认 `./bge-small-zh-v1.5`，可用 `BP_MODEL_PATH` 覆盖）。cross-encoder 重排模型同理（`BP_RERANK_MODEL`）。

## Docker

```bash
# 轻量版：零模型依赖，检索自动降级 TF + BM25
docker compose up

# 完整版：BGE Dense 检索（需先下载模型到 ./bge-small-zh-v1.5）
docker compose --profile full up
```

## 测试与质量

```bash
pytest          # 264 用例，覆盖率 88%（门槛 30%，内置于 pyproject）
ruff check .    # lint
```

- 测试全程零外部依赖：占位 API Key + 强制模型降级路径（`tests/conftest.py`），本地与 CI 行为一致
- 单测驱动修复了 4 处真实缺陷：并行执行器错误吞没、降级检索中文几乎无法命中、图谱深度检查时序、交互式反馈匹配失效

## 生产加固

- **请求限流**：按客户端 IP 的滑动窗口限流（纯标准库实现），`RATE_LIMIT_RPM` 控制每分钟上限（默认 60，0=关闭），`/health` 探针豁免，超限返回 429 + `Retry-After`
- **LLM 重试与降级**：`BP_LLM_MAX_RETRIES` 控制网关层指数退避重试（默认 2，SDK 层已置 0 防两层重试相乘，`BP_LLM_RETRY_BACKOFF` 控制退避基数）；重试耗尽后 SSE 流推送 `error` 事件（"服务暂时不可用"话术），用户端拿到降级答复而非断流
- **优雅关停**：服务 lifespan 关闭时关停工具执行线程池（不阻塞在途任务），避免进程退出挂起

## 医疗合规

`src/compliance.py` 统一合规基建（零第三方依赖），`serve.py` 在创建 SSE 流式响应**之前**完成全部检查——流一旦开始就无法再改状态码，拒绝场景统一返回 400 JSON：

- **免责声明策略**：统一文案常量 `MEDICAL_DISCLAIMER` 三处联动——SSE `done` 事件携带 `disclaimer` 字段、Web UI 页脚常驻展示（标注"以上内容由 AI 生成"）、输出安全闸强制附加在最终回答尾部
- **输入校验规则**（`validate_query`）：空/纯空白输入拒绝、长度上限 500 字（超长拒绝）、控制字符剥离（保留 `\t\n\r`），校验通过后用清洗文本替换原始输入进入 ReAct
- **注入拦截**（`check_injection`）：三类高置信 Prompt 注入模式（指令覆盖中/英文、扮演医务人员的角色劫持、系统提示词探测）命中即 400 拒绝并 `logger.warning` 记录风险类别；只收录高置信攻击语句，宁可漏报也不误伤正常医疗咨询
- **密钥管理**：LLM API Key 等敏感配置仅通过环境变量注入，代码零硬编码；`.env.example` 提供全量变量模板（占位符），`.gitignore` 忽略 `.env` 绝不入库

## 可观测性

`src/observability.py` 提供结构化日志 + Prometheus 指标（P2 基建），随服务自动生效：

- **X-Request-ID 全链路串联**：中间件自动生成 12 位十六进制请求 ID（或透传网关传入值）并回写响应头，同一次请求内的全部日志行携带同一 `request_id`——包括 SSE 流式接口里线程池中执行的 ReAct 各步（contextvars 手动跨线程传播）
- **JSON 行式日志**：默认一行一条 JSON（Docker / Loki 直接解析），`BP_LOG_FORMAT=text` 切人读格式，`BP_LOG_LEVEL` 控制级别（默认 INFO）。真实样例：

```json
{"ts": "2026-10-23T15:30:26", "level": "INFO", "logger": "serve", "msg": "预加载混合检索索引（BM25 + BGE + RRF）...", "request_id": "3f9c2a1b8e07"}
```

- **指标清单**（`GET /metrics`，Prometheus 文本协议，限流豁免）：

| 指标 | 类型 | 标签 | 说明 |
|---|---|---|---|
| `http_requests_total` | Counter | method / path / status | 请求计数（path 取路由模板如 `/session/{session_id}`，防标签基数爆炸） |
| `http_request_duration_seconds` | Histogram | path | 请求延迟（桶 5ms~10s；SSE 流只统计到响应头返回） |
| `rate_limit_rejected_total` | Counter | — | 限流拒绝（429）计数，便于告警配额 |
| `llm_calls_total` | Counter | status | LLM 调用计数（ok / error） |
| `llm_call_duration_seconds` | Histogram | — | LLM 调用延迟（桶 0.1s~60s） |
| `GET /metrics` | 端点 | — | 抓取端点（`include_in_schema=False`，不进 OpenAPI） |

```bash
curl localhost:8000/metrics    # 抓取指标
curl -H "X-Request-ID: abc123" localhost:8000/health    # 响应头原样回传该 ID
```

中间件注册顺序：`install_observability(app)` 在 `install_rate_limit(app)` 之后调用（FastAPI 后注册的 HTTP 中间件位于外层），限流器返回的 429 才会流经可观测中间件被统计进 `rate_limit_rejected_total`；`install_auth(app)` 注册在限流之前（内层），请求顺序为 可观测 → 限流 → 鉴权 → 路由。

## 平台安全与水平扩展（P4）

**双级 API Key 鉴权**（`src/auth.py`）：`BP_API_KEYS`（用户级）/ `BP_ADMIN_KEYS`（管理级）逗号分隔配置，启用后全部业务端点要求 `Authorization: Bearer <key>`；无 Key 或错误 Key 返回 401，用户 Key 访问管理端点返回 403（语义分离）；`/health`、`/metrics`、`/`（Web UI）与 OPTIONS 预检豁免；两变量均未配置时自动放行，开发模式零负担。

```bash
export BP_API_KEYS="user-key-1,user-key-2"    # 用户级（逗号分隔多 Key）
export BP_ADMIN_KEYS="admin-key-1"            # 管理级
# 普通问答：用户 Key 放行
curl -N -H "Authorization: Bearer user-key-1" -d '{"question":"袖带怎么绑"}' localhost:8000/query/manual
# 管理操作：管理 Key 放行并落审计
curl -X POST -H "Authorization: Bearer admin-key-1" localhost:8000/cache/clear
# 用户 Key 打管理端点 → 403
curl -X POST -H "Authorization: Bearer user-key-1" localhost:8000/cache/clear
```

**管理操作审计**（`src/audit.py`）：`/cache/clear` 等管理端点挂 `require_admin` 依赖并逐次调用 `record_audit` 落审计日志——JSONL 追加写入 `logs/audit.jsonl`（`BP_AUDIT_PATH` 可覆盖路径），记录 action / target / detail / 时间戳 / actor；actor 只存 Key 的 SHA-256 前 8 位指纹，明文密钥绝不落盘。

**LLM 统一网关**（`src/llm_gateway.py`）：两版 ReAct（手写版 + FC 版）收敛到 `chat()` 单一入口，模块不再各自持有 SDK 客户端；网关内集中完成 `llm_call_timer` 埋点、显式指数退避重试与**主备链路切换**——主链路 DashScope（`DASHSCOPE_API_KEY` + `AGENT_MODEL`，默认 qwen-max）重试耗尽后自动切备链路（`BP_LLM_FALLBACK_BASE_URL` / `BP_LLM_FALLBACK_API_KEY` 同时配置才启用，`BP_LLM_FALLBACK_MODEL` 缺省沿用主模型名），主备各自再跑完整重试轮，单点供应商故障不再中断 SSE 流。

**会话状态外置**（`src/conversation_store.py`）：`ConversationStore` 抽象基类 + 进程内存实现（默认，零依赖）+ Redis 实现；配置 `BP_REDIS_URL` 即切换 Redis 存储，多副本部署共享会话状态实现水平扩展，连接失败自动回落内存存储并告警（不阻塞启动）。Redis 为可选依赖：

```bash
pip install -e ".[redis]"                       # 安装可选依赖
export BP_REDIS_URL=redis://localhost:6379/0    # 切换外置存储
```

docker-compose 场景在 `services` 下追加 `redis` 服务（`image: redis:7-alpine`），服务环境设 `BP_REDIS_URL=redis://redis:6379/0`，即可多副本共享会话（本地默认单副本内存存储，无需 Redis）。

## 评估门禁

检索质量回归在合入前拦截：`evals/gate.py` 用**与线上一致的语料**（`tools._build_hybrid_docs()`，20 FAQ + 6 产品 + 9 故障共 35 篇）构建 BM25 索引，对 `evals/evalset.json` 黄金集（34 条真实客户口吻中文问句 → 期望命中文档 id，覆盖 FAQ 咨询 / 产品型号 / 故障症状三类，难度混合）统计 **BM25-only 词面检索 Top1 / Top3 命中率**：

- **为什么锁 BM25-only**：纯 numpy 确定性计算，CI 无模型权重可跑；BM25 是混合检索（BM25 + BGE + RRF + Rerank）的召回下限，词面检索劣化必然拖累混合检索——注意这是门禁基线口径，与上方混合检索（含 Dense + Rerank）的完整评测数据口径不同
- **阈值**（`evals/thresholds.json`，实测数字留余量向下取整）：任一项低于阈值则以退出码 1 结束，CI `eval-gate` job 挂红

| 指标 | 实测 | 阈值 |
|---|---|---|
| Top1 命中率 | 94.1%（32/34） | ≥ 90% |
| Top3 命中率 | 100%（34/34） | ≥ 95% |

```bash
python evals/gate.py    # 本地运行：输出每项 PASS/FAIL + 实测数字
```

CI 流水线为 lint → test → **eval-gate** → docker（docker 需前三项全部通过）。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query/manual` | 手写版 ReAct，SSE 流式返回每步 |
| POST | `/query/fc` | Function Calling 版，SSE 流式返回每步 |
| POST | `/session/create` | 创建会话 |
| POST | `/session/clear` | 清空会话历史 |
| GET | `/session/{id}` / `/sessions` | 会话历史 / 会话列表 |
| GET | `/health` | 健康检查 |
| GET | `/cache/stats` | 缓存统计 |
| POST | `/cache/clear` | 清空缓存（**管理端点**：启用鉴权后需管理 Key，见 P4 小节） |

## 项目结构

```
├── src/
│   ├── serve.py                # FastAPI + SSE 流式服务（多轮会话）
│   ├── react_manual.py         # 手写 ReAct（正则解析，过程透明）
│   ├── react_function_calling.py # FC 协议版（tool_calls + 并行执行）
│   ├── tools.py                # 6 工具：FAQ/产品/计算/测量指导/排查/图谱排查
│   ├── hybrid_search.py        # BM25 + BGE + RRF + Rerank 混合检索
│   ├── troubleshoot_graph.py   # 故障决策图谱（BFS 多跳 + 交互式 walk）
│   ├── safety_gate.py          # 输入/输出安全闸（注入/越狱/医学越界/免责声明）
│   ├── guardrails.py           # 状态机 + 调用去重 + 连续失败恢复
│   ├── semantic_cache.py       # 两级语义缓存（问题级 + 工具级）
│   ├── conversation.py         # 会话管理（截断/过期清理，委托 Store）
│   ├── conversation_store.py   # 会话存储抽象：内存（默认）/ Redis 外置（P4）
│   ├── llm_gateway.py          # LLM 统一网关：埋点/重试/主备切换（P4）
│   ├── auth.py                 # 双级 API Key 鉴权（P4）
│   ├── audit.py                # 管理操作审计 JSONL（P4）
│   └── parallel_executor.py    # 并行工具执行器（保序）
├── tests/                      # 264 用例，18 个测试模块
├── evals/                      # 检索评估门禁（evalset.json 黄金集 + thresholds.json 阈值 + gate.py）
├── data/                       # 评估数据集（benchmark_100 / adversarial）
├── scripts/                    # 回归脚本 / 评估集生成
├── index.html                  # Web UI
└── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```

## License

MIT

# 血压计客服 ReAct Agent

[![CI](https://github.com/zhou06080120/blood-pressure-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zhou06080120/blood-pressure-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)](https://github.com/zhou06080120/blood-pressure-agent/actions)

面向医疗器械（血压计）场景的智能客服系统：双实现 ReAct Agent（手写解析版 + Function Calling 协议版）、BM25 + BGE Dense + RRF + Reranker 四层混合检索、故障排查决策图谱（BFS 多跳推演）、六道安全防线与两级语义缓存，通过 FastAPI + SSE 提供流式多轮对话服务。

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
# 编辑 .env 填入 DASHSCOPE_API_KEY（手写版）或 DEEPSEEK_API_KEY（FC 版）
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
pytest          # 190 用例，覆盖率 86%（门槛 30%，内置于 pyproject）
ruff check .    # lint
```

- 测试全程零外部依赖：占位 API Key + 强制模型降级路径（`tests/conftest.py`），本地与 CI 行为一致
- 单测驱动修复了 4 处真实缺陷：并行执行器错误吞没、降级检索中文几乎无法命中、图谱深度检查时序、交互式反馈匹配失效

## 生产加固

- **请求限流**：按客户端 IP 的滑动窗口限流（纯标准库实现），`RATE_LIMIT_RPM` 控制每分钟上限（默认 60，0=关闭），`/health` 探针豁免，超限返回 429 + `Retry-After`
- **LLM 重试与降级**：`BP_LLM_MAX_RETRIES` 控制 SDK 层指数退避重试（默认 2）；重试耗尽后 SSE 流推送 `error` 事件（"服务暂时不可用"话术），用户端拿到降级答复而非断流
- **优雅关停**：服务 lifespan 关闭时关停工具执行线程池（不阻塞在途任务），避免进程退出挂起

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query/manual` | 手写版 ReAct，SSE 流式返回每步 |
| POST | `/query/fc` | Function Calling 版，SSE 流式返回每步 |
| POST | `/session/create` | 创建会话 |
| POST | `/session/clear` | 清空会话历史 |
| GET | `/session/{id}` / `/sessions` | 会话历史 / 会话列表 |
| GET | `/health` | 健康检查 |
| GET | `/cache/stats` / `/cache/clear` | 缓存统计 / 清空 |

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
│   ├── conversation.py         # 会话管理（截断/过期清理）
│   └── parallel_executor.py    # 并行工具执行器（保序）
├── tests/                      # 190 用例，13 个测试模块
├── data/                       # 评估数据集（benchmark_100 / adversarial）
├── scripts/                    # 回归脚本 / 评估集生成
├── index.html                  # Web UI
└── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```

## License

MIT

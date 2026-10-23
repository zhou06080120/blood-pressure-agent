"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI —— 血压计客服

接口（对应 react_financial_agent/src/serve.py）：
  POST /query/manual      - 手写版 ReAct，流式返回每步（支持多轮）
  POST /query/fc          - Function Calling 版，流式返回每步（支持多轮）
  POST /session/create    - 创建新会话，返回 session_id
  POST /session/clear     - 清空指定会话历史
  GET  /session/{id}      - 获取会话历史摘要
  GET  /sessions          - 列出所有会话（管理/调试）
  GET  /health            - 健康检查

多轮对话设计：
  - 前端首次加载时调用 /session/create 拿到 session_id
  - 每次提问把 session_id 随 QueryRequest 一起发送
  - 服务端从 ConversationManager 取历史 history，拼到 system_prompt 后
  - ReAct 循环只在本轮内运行（Thought/Action/Observation 是临时的）
  - 本轮 Final Answer 后，把 user 问题 + assistant 回答存回会话历史
  - 跨轮只带 user/assistant 最终消息，避免上下文膨胀

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

# ═══ 导入区：标准库 + FastAPI/Pydantic ═══
import os  # 环境变量读取（健康检查回报模型名/并行度）
import sys  # 修改 sys.path 以支持从任意目录启动
import json  # SSE 事件 JSON 序列化
import logging  # 服务运行日志
import asyncio  # 异步队列与事件循环
from contextlib import asynccontextmanager  # 把异步生成器注册为应用生命周期钩子
from pathlib import Path  # 定位 index.html 与 src 目录

from fastapi import FastAPI  # Web 框架主体
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # HTML / JSON（合规拒绝）/ SSE 流式响应
from pydantic import BaseModel  # 请求体模型定义与校验

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 faiss 与系统库的 OpenMP 重复加载冲突
sys.path.insert(0, str(Path(__file__).parent))  # 把 src 目录加入模块搜索路径，从项目根启动也能导入各模块

from observability import install_observability, request_id_var, setup_logging  # 可观测基建：日志初始化/指标中间件/请求 ID 上下文（sys.path 注入后才能导入）
from compliance import MEDICAL_DISCLAIMER, ComplianceError, check_injection, validate_query  # 医疗合规：输入校验 / 注入拦截 / 统一免责声明（sys.path 注入后才能导入）

setup_logging()  # 结构化日志初始化：默认 JSON 行式（供日志采集器解析），BP_LOG_FORMAT=text 切人读格式
logger = logging.getLogger(__name__)  # 本模块日志器


# ── 预加载混合检索索引（启动时执行一次）───────────────────────────────────────
@asynccontextmanager  # 将下方异步生成器包装成 lifespan 上下文管理器
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动时异步预加载混合检索索引，避免首个请求冷启动延迟"""
    logger.info("预加载混合检索索引（BM25 + BGE + RRF）...")  # 预热开始
    from tools import _load_hybrid  # 延迟导入：混合检索索引构建函数
    try:  # 预加载可能失败（缺模型/缺文件）
        await asyncio.to_thread(_load_hybrid)  # 丢给工作线程执行，不阻塞事件循环
    except Exception as e:  # 预加载失败不致命
        logger.warning(f"混合检索预加载失败（将在首次调用时重试）: {e}")  # 降级策略：首次请求时再构建
    logger.info("预加载完成，服务就绪")  # 预热完成
    yield  # yield 前=启动逻辑，yield 后=关闭逻辑
    from parallel_executor import shutdown as shutdown_tool_pool  # 工具线程池关停入口（延迟导入：与上方各模块导入风格一致）
    shutdown_tool_pool()  # 优雅关停：释放工具执行线程（不等待在途任务）
    logger.info("工具线程池已关闭，服务退出")  # 记录关停完成


app = FastAPI(title="ReAct 血压计客服 Agent", lifespan=lifespan)  # 创建应用实例并挂载生命周期钩子


# ── 生产加固：请求限流 ────────────────────────────────────────────────────────
def _setup_rate_limit(app: FastAPI) -> None:
    """在应用上注册按客户端 IP 的滑动窗口限流（RATE_LIMIT_RPM<=0 时零开销跳过）"""
    from rate_limit import install_rate_limit  # 延迟导入：src 内平铺模块，与各路由函数内导入风格一致
    install_rate_limit(app)  # 注册 HTTP 中间件：请求进入路由前拦截，超阈值返回 429


_setup_rate_limit(app)  # 应用创建后立即注册（此时读 RATE_LIMIT_RPM 环境变量）

# ── 可观测性：指标中间件 + /metrics 端点 ─────────────────────────────────────
install_observability(app)  # 必须在限流之后注册：FastAPI 后注册的 HTTP 中间件位于外层，限流器返回的 429 才会流经本中间件被统计进指标


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):  # 查询请求体：Pydantic 自动校验字段类型
    """查询请求体：question 为必填问题，session_id 为空表示单轮、有值表示多轮"""
    question:   str  # 用户问题（必填）
    max_steps:  int = 10  # ReAct 循环最大步数，默认 10
    session_id: str | None = None   # 多轮对话：首次为 None，前端应先创建会话


# ── 医疗合规：输入校验 + Prompt 注入拦截 ──────────────────────────────────────
def _compliance_check(question: str) -> tuple[str, JSONResponse | None]:
    """查询入口统一合规检查：先 validate_query 清洗校验，再 check_injection 注入检测。

    必须在创建 StreamingResponse 之前完成——SSE 一旦开始流式输出就无法再改状态码，
    拒绝场景只能靠这里的普通 400 JSON 响应表达。

    :return: (清洗后的安全文本, None) 表示通过；(原始文本, 400 响应) 表示拒绝
    """
    try:
        cleaned = validate_query(question)  # 校验并清洗：空输入 / 超长拒绝，控制字符剥离
    except ComplianceError as e:  # 校验失败：str(e) 即构造时传入的用户可见报错文案
        return question, JSONResponse(status_code=400, content={"detail": str(e)})
    category = check_injection(cleaned)  # 注入检测：命中返回风险类别，未命中返回 None
    if category:  # 命中高置信注入模式：记录风险类别供安全审计，用户侧返回统一话术（不回显原文防二次注入）
        logger.warning(f"检测到 Prompt 注入（风险类别: {category}），已拒绝该请求")  # 风险类别随结构化日志（含 request_id）落盘
        return cleaned, JSONResponse(
            status_code=400,  # 与校验失败同为 400，但 detail 固定为统一安全话术
            content={"detail": "检测到不安全的指令内容，请输入正常的血压计使用或健康咨询问题"})
    return cleaned, None  # 全部通过：返回清洗后的安全文本供 ReAct 使用


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    """将单步事件字典序列化为 SSE data 帧（末尾空行分隔事件）"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"  # SSE 协议格式：data: JSON + 空行分帧


# ═══ 功能块：_stream_react —— 同步 ReAct 生成器 → 工作线程 → asyncio.Queue → SSE 流 ═══
async def _stream_react(question: str, max_steps: int, mode: str,
                        session_id: str | None = None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    多轮对话：若提供 session_id，从 ConversationManager 取历史拼入上下文，
    本轮结束后把 user 问题 + assistant 最终回答写回会话历史。
    """
    if mode == "manual":  # 按前端选择的实现方式导入
        from react_manual import run as react_run  # 手写 Prompt 解析版 ReAct
    else:
        from react_function_calling import run as react_run  # Function Calling 版 ReAct

    # 取多轮对话历史（跨轮只带 user/assistant 最终消息）
    history = []  # 默认空历史（单轮对话）
    if session_id:  # 携带会话 ID 才查历史
        from conversation import conv_manager  # 会话管理器单例
        history = conv_manager.get_history(session_id)  # 取该会话的 user/assistant 消息列表

    queue: asyncio.Queue = asyncio.Queue()  # 跨线程事件队列：worker 线程 put、本协程 get
    _SENTINEL = object()  # 哨兵对象：全局唯一，标记"worker 已跑完"
    rid = request_id_var.get()  # 先在请求协程上下文里取 request_id（线程池线程不继承 contextvars，只能手动带过去）

    def _worker():  # 工作线程函数：同步执行 ReAct 生成器
        token = request_id_var.set(rid)  # 线程池不继承上下文，手动注入使 ReAct 各步日志带 request_id
        try:
            for step_data in react_run(question, max_steps=max_steps, history=history):  # 逐步消费同步生成器
                queue.put_nowait(step_data)  # 每步事件塞进队列
        finally:  # 无论正常结束还是抛异常
            queue.put_nowait(_SENTINEL)  # 必须投递哨兵，否则主协程会永久等待
            request_id_var.reset(token)  # 线程池复用线程，必须复位防串号到下一任务

    yield _sse({"type": "start", "question": question, "mode": mode,  # 先推 start 事件：前端立即显示"思考中"
                "session_id": session_id, "history_turns": len(history) // 2})  # 历史轮数（user+assistant 成对，除 2）

    loop = asyncio.get_event_loop()  # 取当前事件循环
    loop.run_in_executor(None, _worker)  # 把 worker 丢进默认线程池执行（None=默认执行器）

    # 收集本轮最终回答，用于回写会话历史
    final_answer = None  # 本轮最终答案暂存
    while True:  # 事件泵：持续从队列取事件
        step_data = await queue.get()  # 阻塞等待下一条（新事件或哨兵）
        if step_data is _SENTINEL:  # 收到哨兵：worker 已结束
            break  # 退出事件泵
        if step_data.get("type") == "final":  # 捕获最终答案事件
            final_answer = step_data.get("answer", "")  # 暂存用于历史回写
        yield _sse(step_data)  # 原样转发给前端

    # 回写多轮对话历史：user 问题 + assistant 最终回答
    if session_id and final_answer:  # 多轮模式且有答案才回写
        from conversation import conv_manager  # 会话管理器单例
        conv_manager.add_message(session_id, "user", question)  # 存 user 问题
        conv_manager.add_message(session_id, "assistant", final_answer)  # 存 assistant 最终回答

    yield _sse({"type": "done", "disclaimer": MEDICAL_DISCLAIMER})  # 最后推 done 事件：附带统一医疗免责声明，前端结束本轮


# ── 查询路由 ──────────────────────────────────────────────────────────────────
@app.post("/query/manual")  # 注册 POST 路由：手写版查询
async def query_manual(req: QueryRequest):  # 请求体自动解析为 QueryRequest
    """手写 Prompt 解析版 ReAct 的流式查询入口（SSE）"""
    cleaned, rejection = _compliance_check(req.question)  # 合规前置检查（必须在建流前完成，拒绝时才能返回 400 状态码）
    if rejection is not None:  # 校验 / 注入未通过：直接返回 400 JSON，不进入 SSE 流
        return rejection
    return StreamingResponse(  # 返回 SSE 流式响应
        _stream_react(cleaned, req.max_steps, "manual", req.session_id),  # 以 manual 模式运行事件流（用清洗后的安全文本）
        media_type="text/event-stream",  # SSE MIME 类型
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁缓存/禁代理缓冲，保证实时推送
    )


@app.post("/query/fc")  # 注册 POST 路由：FC 版查询
async def query_fc(req: QueryRequest):  # 请求体自动解析为 QueryRequest
    """Function Calling 版 ReAct 的流式查询入口（SSE）"""
    cleaned, rejection = _compliance_check(req.question)  # 合规前置检查（必须在建流前完成，拒绝时才能返回 400 状态码）
    if rejection is not None:  # 校验 / 注入未通过：直接返回 400 JSON，不进入 SSE 流
        return rejection
    return StreamingResponse(  # 返回 SSE 流式响应
        _stream_react(cleaned, req.max_steps, "fc", req.session_id),  # 以 fc 模式运行事件流（用清洗后的安全文本）
        media_type="text/event-stream",  # SSE MIME 类型
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁缓存/禁代理缓冲，保证实时推送
    )


# ── 会话管理路由 ─────────────────────────────────────────────────────────────
@app.post("/session/create")  # 注册 POST 路由：创建会话
async def create_session():
    """创建新会话，返回 session_id"""
    from conversation import conv_manager  # 会话管理器单例
    # 顺带清理过期会话
    cleaned = conv_manager.cleanup_expired()  # 清理过期会话，返回清理数量
    if cleaned:  # 有清理才记日志
        logger.info(f"清理过期会话 {cleaned} 个")  # 记录清理数量
    session_id = conv_manager.create_session()  # 创建新会话拿 ID
    return {"session_id": session_id, "message_count": 0}  # 返回会话 ID 与初始消息数


@app.post("/session/clear")  # 注册 POST 路由：清空会话
async def clear_session(req: dict):  # 请求体为原始 dict
    """清空指定会话历史（保留 session_id）"""
    from conversation import conv_manager  # 会话管理器单例
    session_id = req.get("session_id")  # 取目标会话 ID
    if not session_id:  # 缺参校验
        return {"ok": False, "error": "缺少 session_id"}  # 报错返回
    ok = conv_manager.clear(session_id)  # 清空历史（会话保留）
    return {"ok": ok}  # 返回清空结果


@app.get("/session/{session_id}")  # 注册 GET 路由：查询单个会话（路径参数）
async def get_session(session_id: str):  # 路径参数自动注入
    """获取会话历史摘要"""
    from conversation import conv_manager  # 会话管理器单例
    info = conv_manager.get_info(session_id)  # 查会话摘要
    if info is None:  # 会话不存在
        return {"ok": False, "error": f"会话 {session_id} 不存在"}  # 报错返回
    return {"ok": True, **info}  # 展开摘要字段返回


@app.get("/sessions")  # 注册 GET 路由：列出全部会话
async def list_sessions():
    """列出所有会话（管理/调试用）"""
    from conversation import conv_manager  # 会话管理器单例
    return {"sessions": conv_manager.list_sessions()}  # 返回会话列表


@app.get("/health")  # 注册 GET 路由：健康检查
async def health():
    """健康检查：返回服务状态、模型名、并行度与缓存统计"""
    from semantic_cache import cache_stats  # 缓存统计函数
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max"),  # 服务状态与当前模型
            "parallel_workers": os.getenv("BP_PARALLEL_WORKERS", "4"),  # 并行执行器工作线程数
            "cache": cache_stats()}  # 语义缓存命中率统计


@app.get("/cache/stats")  # 注册 GET 路由：缓存统计
async def cache_stats_endpoint():
    """语义缓存命中率统计（问题级 + 工具级）"""
    from semantic_cache import cache_stats  # 缓存统计函数
    return {"ok": True, **cache_stats()}  # 展开两级缓存统计返回


@app.post("/cache/clear")  # 注册 POST 路由：清空缓存
async def cache_clear_endpoint():
    """清空语义缓存（问题级 + 工具级）"""
    from semantic_cache import question_cache, tool_cache  # 两级缓存单例
    question_cache._entries.clear()  # 清空问题级缓存条目
    question_cache._hits = question_cache._misses = question_cache._semantic_hits = 0  # 问题级计数器归零
    tool_cache._entries.clear()  # 清空工具级缓存条目
    tool_cache._hits = tool_cache._misses = 0  # 工具级计数器归零
    return {"ok": True}  # 返回成功


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"  # 前端页面路径：项目根目录的 index.html

@app.get("/")  # 注册根路由
async def root():
    """首页：托管项目根目录的 index.html（前端 Web UI）"""
    if HTML_PATH.exists():  # 页面文件存在
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))  # 以 UTF-8 读出并返回 HTML
    return HTMLResponse("<h2>index.html not found</h2>")  # 兜底提示

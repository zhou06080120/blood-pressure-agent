"""
Function Calling API 版 ReAct Agent —— 血压计客服

教学重点（对应 react_financial_agent/src/react_function_calling.py）：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "血压计测出来的值偏高怎么办？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers
  export DASHSCOPE_API_KEY="sk-xxx"
"""

# ═══ 导入区：OpenAI SDK + 标准库 ═══
import os  # 环境变量读取与 KMP 开关
import json  # 解析 tool_calls 的参数 JSON 字符串
import time  # 计时
import logging  # 日志
import argparse  # CLI 参数
from typing import Generator  # run() 是生成器类型

from openai import OpenAI  # 兼容 OpenAI 协议的 SDK（DeepSeek/通义均可复用）

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许多份 OpenMP 共存，防 numpy/faiss 崩溃

logging.basicConfig(level=logging.WARNING)  # 日志级别压到警告
logger = logging.getLogger(__name__)  # 模块日志器

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
client = OpenAI(  # 创建 LLM 客户端（FC 版默认走 DeepSeek）
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量取 API Key
    base_url="https://api.deepseek.com",  # DeepSeek 的 OpenAI 兼容端点
    max_retries=int(os.getenv("BP_LLM_MAX_RETRIES", "2")),  # SDK 内置指数退避重试次数（环境变量可覆盖，0=不重试）
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")  # 模型名可用 AGENT_MODEL 覆盖

# FC 版系统提示词：约束角色/规则/工具使用优先级（与手写版内容对齐，但不需要输出格式约定——格式由 API 处理）
FC_SYSTEM_PROMPT = """你是一个专业的医疗器械客服助手，专门为血压计APP用户提供使用咨询、测量指导和故障排查服务。
规则：
- 查询具体型号参数前，必须先用 product_lookup 获取产品规格
- 数字计算（如平均动脉压、脉压差）必须使用 calculator 工具，不能心算
- 故障排查优先使用 graph_troubleshoot 获得分步排查决策路径，troubleshoot 作为补充
- 同一轮内相互独立的多个工具调用请一次性全部给出（支持并行执行），避免逐个等待
- Final Answer 必须引用具体数据来源
- 回答应专业、准确、易懂，必要时给出分步骤操作指引
- 医学建议仅供参考，不可替代医生诊断，涉及诊断和治疗建议提醒用户就医
- 如果没有合适工具能回答，直接说明原因
"""


# ═══ 功能块：FC 版 ReAct 主循环（生成器） —— 六道防线串联的完整执行流 ═══
def run(question: str, max_steps: int = 10,
        history: list[dict] | None = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环（v2：状态机 + 幻觉自纠错 + 并行工具 + 语义缓存 + 安全闸）

    新增保障（相对 v1）：
      1. 输入安全闸：Prompt 注入/越狱/医学诊断开药请求 → 拒绝并返回安全应答
      2. 问题级语义缓存：高频问题命中直接返回（0 次 LLM 调用）
      3. 决策状态机：限制最大循环深度，杜绝无限往返
      4. 工具调用去重：相同(工具+参数)重复调用自动阻断并注入纠错提示
      5. 并行工具执行：同一轮多个 tool_calls 并发执行（ThreadPoolExecutor）
      6. 输出安全闸：Final Answer 强制附加医学免责声明

    yield 格式与 v1 保持兼容（type: action/final/max_steps），新增扩展字段：
      "guards": 状态机快照  "cached": 缓存命中类型  "parallel": 本轮并行工具数
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA  # 工具注册表 + JSON Schema（FC 协议需要）
    from safety_gate import check_input, enforce_disclaimer  # 输入闸 / 输出闸
    from guardrails import make_guards  # 状态机 + 调用监视器
    from semantic_cache import question_cache, tool_cache  # 两级语义缓存
    from parallel_executor import run_parallel  # 并行执行器

    # ── 防线 1：输入安全闸 ──
    gate = check_input(question)  # 检查是否注入/越狱/医学越界
    if gate["blocked"]:  # 命中拦截规则
        yield {"step": 0, "type": "final", "thought": "",
               "answer": gate["response"], "safety_blocked": gate["reason"]}  # 不进 LLM，直接 yield 安全应答并标记拦截原因
        return  # 终止生成器

    # ── 防线 2：问题级语义缓存（GPTCache 风格） ──
    cached = question_cache.get(question)  # 查问题缓存
    if cached is not None:  # 命中（exact 或 semantic）
        answer, hit_type = cached  # 拆出答案与命中类型
        yield {"step": 0, "type": "final", "thought": "",
               "answer": enforce_disclaimer(answer),  # 缓存答案也要过输出闸补免责声明
               "cached": hit_type}  # 标记缓存命中类型
        return

    sm, monitor = make_guards()  # 创建状态机与调用监视器
    sm.transition(sm.PLANNING)  # INIT → PLANNING：开始规划

    messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]  # 消息列表：系统提示打头
    # 拼接多轮对话历史（user/assistant 交替，仅含最终回答）
    if history:  # 有多轮历史
        messages.extend(history)  # 追加历史消息（不含中间步骤）
    messages.append({"role": "user", "content": question})  # 追加本轮用户问题

    for step in range(1, max_steps + 1):  # 主循环：最多 max_steps 轮
        # ── 防线 3：状态机最大循环深度 ──
        if sm.loop_exhausted:  # 深度耗尽
            yield {"step": step, "type": "max_steps",
                   "answer": (f"已达最大循环深度 {sm.max_depth}，为避免无限往返已收敛。"
                              "请补充更具体的信息（症状/型号/错误码）后重新提问。"),  # 兜底收敛提示
                   "guards": sm.snapshot()}  # 携带状态机快照
            return

        try:  # LLM 调用加固：网络抖动/超时/SDK 重试耗尽不让 SSE 流直接断掉
            response = client.chat.completions.create(  # 调 LLM：带工具 Schema 的对话补全
                model=MODEL,  # 模型名
                messages=messages,  # 当前对话上下文
                tools=TOOLS_SCHEMA,  # 声明可调用的工具（FC 协议核心）
                tool_choice="auto",  # 由模型自主决定调工具还是直接回答
                temperature=0,  # 温度 0：结果稳定可复现
                timeout=float(os.getenv("BP_LLM_TIMEOUT", "60")),  # 超时（环境变量可调）
            )
        except Exception as e:  # 调用层任何异常（SDK 重试后仍失败）
            logger.warning(f"LLM 调用失败: {e}")  # 记日志（SDK 异常不含密钥）
            yield {"step": step, "type": "error",  # 沿用项目现有 error 事件格式
                   "answer": "服务暂时不可用，请稍后重试或联系人工客服。",  # 给用户的降级话术
                   "error": str(e)[:200]}  # 错误摘要（截断防刷屏）
            return  # 终止生成器：前端收到 error 事件后正常收尾
        msg    = response.choices[0].message  # 本轮消息对象（可能含 tool_calls）
        reason = response.choices[0].finish_reason  # 结束原因：stop=直接回答 / tool_calls=请求调工具

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:  # 给出最终答案
            raw = msg.content or "（模型返回空内容）"  # 空答案兜底文案
            # ── 防线 6：输出安全闸（免责声明强制） ──
            answered = enforce_disclaimer(raw)  # 检查幻觉开药 + 补免责声明
            question_cache.put(question, raw)  # 未加声明的原文入缓存（声明在输出时统一加）
            sm.transition(sm.FINAL)  # 状态转 FINAL
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",  # FC 版 Thought 在模型内部不可见
                "answer": answered,  # 净化+带声明的最终答案
                "cached": None,  # 本轮非缓存命中
                "guards": sm.snapshot(),  # 状态机快照
            }
            return

        # 模型请求调用工具
        messages.append(msg)  # 把含 tool_calls 的助手消息入上下文（协议要求必须回填）
        sm.transition(sm.EXECUTING)  # 状态转 EXECUTING（循环深度 +1）

        # ── 防线 4：工具调用去重（防重复调用同一工具原地打转） ──
        prepared = []          # 待执行
        for tool_call in msg.tool_calls:  # 逐个处理本轮的 tool_calls
            tool_name = tool_call.function.name  # 工具名
            try:
                tool_args = json.loads(tool_call.function.arguments)  # 参数是 JSON 字符串 → 解析为字典
            except json.JSONDecodeError:  # 模型给了非法 JSON
                tool_args = {}  # 降级为空参数

            tool_fn = TOOLS_MAP.get(tool_name)  # 查工具注册表
            if tool_fn is None:  # 不认识的工具名
                observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"  # 观察写成引导性提示
                monitor.record(tool_name, tool_args, observation)  # 记入监视器（计一次失败）
                _append_tool_msg(messages, tool_call, observation)  # 回填 tool 消息
                yield _action_event(step, tool_name, tool_args, observation,
                                    sm, parallel=len(msg.tool_calls))  # yield 本步事件
                continue  # 处理下一个 tool_call

            blocked, hint = monitor.check_before(tool_name, tool_args)  # 去重检查
            if blocked:  # 完全相同的调用已超上限
                observation = hint   # 注入纠错提示代替执行
                monitor.record(tool_name, tool_args, observation)  # 记录
                _append_tool_msg(messages, tool_call, observation)  # 回填
                yield _action_event(step, tool_name, tool_args, observation,
                                    sm, parallel=len(msg.tool_calls),
                                    duplicate_blocked=True)  # 事件带去重阻断标记
                continue

            # 工具级缓存（确定性工具结果复用）
            cached_obs = tool_cache.get(tool_name, tool_args)  # 查工具缓存
            if cached_obs is not None:  # 命中
                observation = str(cached_obs)  # 复用缓存结果
                monitor.record(tool_name, tool_args, observation)  # 记录
                _append_tool_msg(messages, tool_call, observation)  # 回填
                yield _action_event(step, tool_name, tool_args, observation,
                                    sm, parallel=len(msg.tool_calls),
                                    tool_cached=True)  # 事件带工具缓存标记
                continue

            prepared.append((tool_call, tool_name, tool_args, tool_fn))  # 真正需要执行的调用进入待执行队列

        # ── 防线 5：并行执行无依赖工具调用 ──
        if prepared:  # 有待执行调用
            results = run_parallel([  # 交给线程池并发执行
                {"name": name, "fn": fn, "args": args}
                for _, name, args, fn in prepared
            ])
            for (tool_call, tool_name, tool_args, tool_fn), res in zip(prepared, results):  # 结果与调用按序对齐
                observation = res["observation"]  # 工具观察
                # 失败自纠错：记录 + 结果回填给模型自行调整
                monitor.record(tool_name, tool_args, observation)  # 记入监视器（含失败计数）
                if res["error"] is None:  # 执行成功
                    tool_cache.put(tool_name, tool_args, observation)  # 成功结果才入工具缓存
                _append_tool_msg(messages, tool_call, observation)  # 回填 tool 消息
                yield _action_event(step, tool_name, tool_args, observation,
                                    sm, parallel=len(prepared),
                                    elapsed_ms=res["elapsed_ms"])  # 事件带并行数与耗时

        sm.transition(sm.OBSERVING)  # 状态转 OBSERVING：本轮观察完成

        # ── 连续失败策略调整（自纠错注入） ──
        hint = monitor.recovery_hint()  # 连续失败达阈值则取恢复提示
        if hint:  # 需要策略切换
            sm.transition(sm.RECOVERING)  # 转 RECOVERING
            messages.append({"role": "user", "content": f"[系统纠错提示] {hint}"})  # 以用户消息注入纠错指令
        else:
            sm.transition(sm.PLANNING)  # 正常回到 PLANNING 继续

    yield {  # 主循环走满仍未拿到 final
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ═══ 功能块：工具消息回填 —— 按 OpenAI 协议把观察结果写回对话 ═══
def _append_tool_msg(messages: list, tool_call, observation: str):
    """把工具执行结果以 OpenAI tool 消息格式回填到对话，带 tool_call_id 对齐"""
    messages.append({
        "role":         "tool",  # 角色为 tool
        "tool_call_id": tool_call.id,  # 必须与请求里的 tool_call.id 对齐，否则 API 报错
        "content":      str(observation),  # 观察文本
    })


# ═══ 功能块：事件构造器 —— 统一拼装一次工具调用的 action 事件 ═══
def _action_event(step: int, tool_name: str, tool_args: dict,
                  observation: str, sm, **extra) -> dict:
    """构造一次工具调用的 action 事件字典（Thought 在 FC 版不可见，固定为空）"""
    ev = {
        "step":         step,  # 步骤号
        "type":         "action",  # 事件类型
        "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
        "action":       tool_name,  # 工具名
        "action_input": tool_args,  # 工具参数
        "observation":  str(observation),  # 观察结果
        "guards":       sm.snapshot(),  # 状态机快照
    }
    ev.update(extra)  # 合并扩展字段（parallel/duplicate_blocked/tool_cached/elapsed_ms）
    return ev  # 返回事件


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {  # ANSI 颜色码表（终端彩色打印）
    "thought": "\033[36m",  # 思考：青色
    "action":  "\033[33m",  # 动作：黄色
    "obs":     "\033[32m",  # 观察：绿色
    "final":   "\033[35m",  # 答案：品红
    "error":   "\033[31m",  # 错误：红色
    "reset":   "\033[0m",  # 重置色
}

def _c(color: str, text: str) -> str:
    """给终端文本包裹 ANSI 颜色码（按 COLORS 键取值）"""
    return f"{COLORS[color]}{text}{COLORS['reset']}"  # 颜色前缀 + 文本 + 重置后缀


# ═══ 功能块：CLI 入口 —— 逐步彩色打印 ReAct 过程 ═══
def run_and_print(question: str, max_steps: int = 10,
                  history: list[dict] | None = None):
    """CLI 入口：运行 Function Calling 版 ReAct 并逐步彩色打印调用过程与最终答案"""
    print(f"\n{'='*60}")  # 分隔线
    print(f"问题: {question}")  # 打印问题
    print(f"模型: {MODEL}  实现: Function Calling")  # 打印模型与实现方式
    if history:  # 有历史
        print(f"多轮对话: 已带入 {len(history)} 条历史消息")  # 提示历史条数
    print('='*60)  # 分隔线

    start = time.time()  # 计时起点

    for step_data in run(question, max_steps=max_steps, history=history):  # 消费生成器
        stype = step_data["type"]  # 事件类型

        if stype == "action":  # 工具调用事件
            print(f"\n[Step {step_data['step']}]")  # 打印步骤标题
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))  # 说明性占位
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))  # 工具名
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))  # 参数
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))  # 观察（截断 300 字）

        elif stype == "final":  # 最终答案事件
            elapsed = time.time() - start  # 总耗时
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))  # 彩色答案
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")  # 步数与耗时

        elif stype in ("error", "max_steps"):  # 异常/步数耗尽事件
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))  # 红色警示


if __name__ == "__main__":  # 直接运行本文件才走 CLI
    parser = argparse.ArgumentParser()  # 参数解析器
    parser.add_argument("--question",  default="血压计袖带应该怎么绑？测出来的值才准？")  # 默认演示问题
    parser.add_argument("--max_steps", type=int, default=10)  # 最大步数
    args = parser.parse_args()  # 解析参数
    run_and_print(args.question, args.max_steps)  # 执行并打印

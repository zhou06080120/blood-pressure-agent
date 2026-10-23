"""
手写 Prompt 解析版 ReAct Agent —— 血压计客服

教学重点（对应 react_financial_agent/src/react_manual.py）：
  1. ReAct 核心循环：Thought → Action → Observation，逐步推理
  2. System Prompt 约束输出格式，Python 正则解析每一步
  3. 对话历史拼接方式：每轮结果追加到 prompt，形成上下文记忆
  4. 停止条件：模型输出 Final Answer 或达到最大步数

使用方式：
  python react_manual.py
  python react_manual.py --question "血压计怎么绑袖带？"
  python react_manual.py --question "..." --max_steps 8 --verbose

依赖：
  pip install openai faiss-cpu sentence-transformers
  export DASHSCOPE_API_KEY="sk-xxx"
"""

# ═══ 导入区：标准库 + OpenAI SDK ═══
import os  # 读取环境变量（API Key、模型名、超时等配置）
import re  # 正则表达式：解析 LLM 输出的 Thought/Action/Final Answer
import json  # 解析 Action Input 的 JSON 参数体
import time  # 统计 ReAct 端到端耗时
import logging  # 日志记录
import argparse  # 命令行参数解析（--question / --max_steps）
from typing import Generator  # 类型标注：run() 是一个生成器

from openai import OpenAI  # OpenAI 兼容客户端（本项目指向 DashScope）

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 faiss 与系统库的 OpenMP 重复加载冲突

logging.basicConfig(level=logging.WARNING)  # 只输出警告及以上，避免工具内部日志刷屏
logger = logging.getLogger(__name__)  # 本模块日志器

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(  # 创建 LLM 客户端（DashScope 的 OpenAI 兼容端点）
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里 DashScope 兼容模式地址
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")  # 默认模型 qwen-max，可用环境变量 AGENT_MODEL 覆盖
# ↓↓↓ DeepSeek 备选客户端配置（原样保留备用，切换供应商时注释上方 client 并取消此处注释）
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


# ── System Prompt ─────────────────────────────────────────────────────────────
# 系统提示词：定义角色、6 个工具用途、严格输出格式与业务规则；
# 三引号内部是发给 LLM 的 prompt 模板原文，为保持 prompt 原样不在其中插注释。
SYSTEM_PROMPT = """你是一个专业的医疗器械客服助手，专门为血压计APP用户提供使用咨询、测量指导和故障排查服务。

工具列表：
1. faq_search(query) - 在血压计知识库（FAQ/产品规格/故障手册）中混合检索（BM25+语义向量，型号与专有名词召回更准）
2. product_lookup(model) - 将血压计型号名称转换为产品规格（品牌/类型/袖带范围/供电/存储/功能）
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数，用于平均动脉压MAP、脉压差等计算）
4. measurement_guide(topic) - 获取血压测量标准操作指南（测量前准备/袖带绑法/测量姿势/测量次数/左右手选择/血压分级/平均动脉压/脉压差）
5. troubleshoot(symptom) - 根据故障症状返回可能原因和解决方案
6. graph_troubleshoot(symptom) - 基于故障排查决策图谱输出分步排查路径（优先使用，可多跳深入排查）

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 查询具体型号参数前，必须先用 product_lookup 获取产品规格
- 数字计算（如平均动脉压、脉压差）必须用 calculator，不能心算
- 故障排查优先使用 graph_troubleshoot 获取分步排查路径
- Final Answer 必须引用具体数据来源（FAQ条目编号/产品规格/医学标准指南）
- 回答应专业、准确、易懂，必要时给出分步骤操作指引
- 医学建议仅供参考，不可替代医生诊断，涉及诊断和治疗建议提醒用户就医
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
- 禁止重复用相同参数调用同一个工具；工具连续失败时应切换策略而非重试
"""

# ── 格式解析 ──────────────────────────────────────────────────────────────────
# 四个预编译正则：分别抽取 Thought / Action / Action Input / Final Answer 字段
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)  # 思考内容：截到 Action/Final Answer 行或文末（DOTALL 支持跨行）
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")  # 工具名：一个单词
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)  # 参数体：最短匹配的 {...} JSON 串
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)  # 最终答案：Final Answer: 之后的全部内容


# ═══ 功能块：_parse_step —— 用正则把 LLM 单步输出解析成结构化事件 ═══
def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)  # 先找 Final Answer：命中说明推理可以收尾
    if final:  # —— 最终答案分支 ——
        thought_m = _THOUGHT_RE.search(text)  # 顺带提取收尾前的 Thought（可能没有）
        return {  # 返回 final 事件
            "type":    "final",  # 事件类型：最终答案
            "thought": thought_m.group(1).strip() if thought_m else "",  # 收尾思考（去首尾空白，无则空串）
            "answer":  final.group(1).strip(),  # 答案正文
        }

    thought_m = _THOUGHT_RE.search(text)  # 提取 Thought
    action_m  = _ACTION_RE.search(text)  # 提取工具名
    input_m   = _ACTION_INPUT_RE.search(text)  # 提取参数体

    if not action_m:  # 既没有 Final 也没有 Action：格式完全不合规
        return {"type": "unparseable", "raw": text}  # 标记不可解析，交由上层自纠错重试

    try:  # 尝试把参数体从 JSON 解析为 dict
        action_input = json.loads(input_m.group(1)) if input_m else {}  # 没写参数就给空 dict
    except json.JSONDecodeError:  # JSON 非法（引号缺失等）
        action_input = {}  # 降级为空参数，避免整个循环崩溃

    return {  # 返回 action 事件
        "type":         "action",  # 事件类型：工具调用
        "thought":      thought_m.group(1).strip() if thought_m else "",  # 本步思考
        "action":       action_m.group(1).strip(),  # 工具名
        "action_input": action_input,  # 工具参数 dict
    }


# ── ReAct 核心循环 ─────────────────────────────────────────────────────────────

# ═══ 功能块：run —— 手写 ReAct 主循环生成器（串联安全闸/缓存/状态机/去重全链路） ═══
def run(question: str, max_steps: int = 10, verbose: bool = True,
        history: list[dict] | None = None) -> Generator[dict, None, None]:
    """
    执行手写 ReAct 循环（v2：安全闸 + 语义缓存 + 状态机 + 去重自纠错）

    每个 yield 的 dict 格式与 v1 兼容：
      {"step": int, "thought": str, "action": str, "action_input": dict, "observation": str}
    最后一个 yield：
      {"step": int, "thought": str, "type": "final", "answer": str}

    相对 v1 的增强：
      - 输入安全闸拦截注入/越狱/诊断开药请求
      - 问题级语义缓存命中直接返回
      - 格式解析失败自动纠正重试（不再直接崩溃）
      - 相同(工具+参数)重复调用阻断 + 连续失败策略调整提示
      - final 答案强制附加医学免责声明
    """
    from tools import TOOLS_MAP  # 延迟导入工具注册表（避免循环依赖、加快模块加载）
    from safety_gate import check_input, enforce_disclaimer  # 输入安全闸 + 免责声明强制
    from guardrails import make_guards  # 状态机 + 工具调用监控器
    from semantic_cache import question_cache, tool_cache  # 问题级/工具级语义缓存

    # 输入安全闸
    gate = check_input(question)  # 第一道防线：检测提示注入/越狱/诊断开药类危险请求
    if gate["blocked"]:  # 命中黑名单
        yield {"step": 0, "type": "final", "thought": "",  # 不进 LLM，直接产出拦截事件
               "answer": gate["response"], "safety_blocked": gate["reason"]}  # 拦截话术 + 拦截原因
        return  # 终止生成器

    # 问题级语义缓存
    cached = question_cache.get(question)  # 第二道防线：先查精确/语义缓存
    if cached is not None:  # 命中缓存（0 步直达答案）
        answer, hit_type = cached  # 拆出缓存答案与命中类型（exact/semantic）
        yield {"step": 0, "type": "final", "thought": "",  # 缓存命中事件
               "answer": enforce_disclaimer(answer), "cached": hit_type}  # 缓存答案也要补免责声明
        return

    sm, monitor = make_guards()  # 第三/四道防线：AgentStateMachine + ToolCallMonitor
    sm.transition(sm.PLANNING)  # 状态机进入 PLANNING：开始规划第一步

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # 消息列表以系统提示词开头
    # 拼接多轮对话历史（user/assistant 交替）
    if history:  # 有多轮历史则追加（跨轮只带最终 user/assistant 消息，防上下文膨胀）
        messages.extend(history)
    messages.append({"role": "user", "content": question})  # 追加本轮用户问题

    format_retries = 0  # 格式解析失败已重试次数
    MAX_FORMAT_RETRY = 2  # 最多自纠错 2 次，防止无限重试

    for step in range(1, max_steps + 1):  # —— ReAct 主循环：最多 max_steps 步 ——
        if sm.loop_exhausted:  # 第五道防线：状态机判定循环深度耗尽
            yield {"step": step, "type": "max_steps",  # 产出收敛终止事件
                   "answer": (f"已达最大循环深度 {sm.max_depth}，为避免无限往返已收敛。"
                              "请补充更具体的信息后重新提问。"),  # 引导用户补充信息
                   "guards": sm.snapshot()}  # 附状态机快照便于排查
            return

        response = client.chat.completions.create(  # 调 LLM 生成下一步
            model=MODEL,  # 模型名
            messages=messages,  # 完整上下文（系统提示 + 历史 + 本轮）
            temperature=0,  # 温度 0：客服场景要求输出稳定、可解析
            stop=["Observation:"],  # 让模型停在调用工具前
            timeout=float(os.getenv("BP_LLM_TIMEOUT", "60")),  # 请求超时秒数，防网络卡死
        )
        llm_output = response.choices[0].message.content.strip()  # 取出模型文本输出并去空白
        parsed = _parse_step(llm_output)  # 正则解析为结构化事件

        if parsed["type"] == "final":  # —— 模型给出最终答案 ——
            answer = enforce_disclaimer(parsed["answer"])  # 第六道防线：强制附加医学免责声明
            question_cache.put(question, parsed["answer"])  # 原始答案写入语义缓存供下次命中
            sm.transition(sm.FINAL)  # 状态机迁移到 FINAL
            yield {  # 产出最终答案事件
                "step":    step,  # 步数
                "type":    "final",  # 事件类型
                "thought": parsed["thought"],  # 收尾思考
                "answer":  answer,  # 含免责声明的答案
                "guards":  sm.snapshot(),  # 状态机快照
            }
            return  # 正常结束生成器

        if parsed["type"] == "unparseable":  # —— 格式解析失败 ——
            # 自纠错：格式解析失败 → 注入纠正提示重试，而非直接崩溃
            if format_retries < MAX_FORMAT_RETRY:  # 还有重试额度
                format_retries += 1  # 消耗一次额度
                messages.append({"role": "assistant", "content": llm_output})  # 把坏输出记为上下文
                messages.append({"role": "user", "content":  # 注入格式纠正指令
                    "你的输出格式不符合要求。必须严格按以下格式：\n"
                    "Thought: 分析\nAction: 工具名\nAction Input: {\"参数\": \"值\"}\n"
                    "或直接输出 'Thought: ...\\nFinal Answer: ...'。请重新输出。"})
                continue  # 回到循环顶部重新请求 LLM
            yield {  # 额度耗尽：产出 error 事件
                "step":        step,  # 步数
                "type":        "error",  # 事件类型
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",  # 截取前 200 字符供排查
            }
            return  # 放弃并终止

        # 执行工具（含去重阻断）
        tool_name  = parsed["action"]  # 模型选择的工具名
        tool_args  = parsed["action_input"]  # 工具参数
        tool_fn    = TOOLS_MAP.get(tool_name)  # 从注册表取工具函数

        if tool_fn is None:  # 工具名不在注册表
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"  # 回传可用清单引导纠错
        else:  # 工具存在
            blocked, hint = monitor.check_before(tool_name, tool_args)  # 第四道防线：重复调用/连续失败检查
            if blocked:  # 被阻断
                observation = hint  # 阻断重复调用，注入纠错提示
            else:  # 允许执行
                cached_obs = tool_cache.get(tool_name, tool_args)  # 工具级语义缓存查询
                if cached_obs is not None:  # 命中工具缓存
                    observation = str(cached_obs)  # 直接复用历史结果，省一次工具执行
                else:  # 未命中：真正执行工具
                    try:  # 执行并写缓存
                        observation = tool_fn(**tool_args)  # 解包参数调用工具
                        tool_cache.put(tool_name, tool_args, observation)  # 结果写入工具缓存
                    except TypeError as e:  # 参数签名不匹配
                        observation = f"工具参数错误: {e}"
                    except Exception as e:  # 其他异常兜底（不中断循环）
                        observation = f"工具执行异常: {e}"

        monitor.record(tool_name, tool_args, str(observation))  # 登记本次调用：供去重指纹与失败统计

        step_result = {  # 组装本步 action 事件
            "step":         step,  # 步数
            "type":         "action",  # 事件类型
            "thought":      parsed["thought"],  # 思考
            "action":       tool_name,  # 工具名
            "action_input": tool_args,  # 参数
            "observation":  str(observation),  # 观察结果（统一转字符串）
            "guards":       sm.snapshot(),  # 状态机快照
        }
        yield step_result  # 推给调用方（CLI 打印 / 服务端 SSE）

        # 将本步结果追加到对话历史
        messages.append({"role": "assistant", "content": llm_output})  # 模型的 Thought+Action 作为 assistant 消息
        messages.append({  # 工具结果伪装成下一条 user 消息回填
            "role":    "user",
            "content": f"Observation: {observation}\n",
        })

        # 连续失败策略调整（自纠错注入）
        hint = monitor.recovery_hint()  # 检测连续失败，生成策略调整提示
        if hint:  # 触发恢复模式
            sm.transition(sm.RECOVERING)  # 进入 RECOVERING 状态
            messages.append({"role": "user", "content": f"[系统纠错提示] {hint}"})  # 注入策略调整提示
            sm.transition(sm.PLANNING)  # 回到 PLANNING 继续规划
        else:  # 正常流转：OBSERVING → PLANNING
            sm.transition(sm.OBSERVING)  # 已拿到观察结果
            sm.transition(sm.PLANNING)  # 回到规划态

    # 超出最大步数，强制终止
    yield {  # 循环走完仍无 Final：产出 max_steps 兜底事件
        "step":   max_steps + 1,  # 步数标记为 max_steps+1
        "type":   "max_steps",  # 事件类型
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",  # 兜底话术
    }


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {  # ANSI 颜色码表：终端彩色打印各事件（原有行尾注释保留）
    "thought":  "\033[36m",   # cyan
    "action":   "\033[33m",   # yellow
    "obs":      "\033[32m",   # green
    "final":    "\033[35m",   # magenta
    "error":    "\033[31m",   # red
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    """给终端文本包裹 ANSI 颜色码（按 COLORS 键取值）"""
    return f"{COLORS[color]}{text}{COLORS['reset']}"  # 颜色前缀 + 内容 + 复位码


# ═══ 功能块：run_and_print —— CLI 入口：消费事件流并彩色打印全过程 ═══
def run_and_print(question: str, max_steps: int = 10,
                  history: list[dict] | None = None):
    """CLI 入口：运行手写 ReAct 并逐步彩色打印 Thought/Action/Observation 与最终答案"""
    print(f"\n{'='*60}")  # 顶部横幅
    print(f"问题: {question}")  # 打印问题
    print(f"模型: {MODEL}  实现: 手写Prompt解析")  # 打印模型与实现方式
    if history:  # 多轮模式
        print(f"多轮对话: 已带入 {len(history)} 条历史消息")  # 提示带入的历史条数
    print('='*60)  # 横幅收尾

    start = time.time()  # 记录开始时间
    step_count = 0  # 事件计数器

    for step_data in run(question, max_steps=max_steps, history=history):  # 逐事件消费生成器
        step_count += 1  # 累计事件数
        stype = step_data["type"]  # 取事件类型分发

        if stype == "action":  # 工具调用事件
            print(f"\n[Step {step_data['step']}]")  # 步骤标题
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))  # 思考（青色）
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))  # 工具名（黄色）
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))  # 参数 JSON
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))  # 观察结果截断 300 字符（绿色）

        elif stype == "final":  # 最终答案事件
            elapsed = time.time() - start  # 计算总耗时
            print(f"\n{'─'*60}")  # 结果分隔线
            if step_data.get("thought"):  # 有收尾思考才打印
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))  # 收尾思考
            print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))  # 最终答案（品红）
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")  # 步数与耗时统计

        elif stype in ("error", "max_steps"):  # 错误/步数耗尽事件
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))  # 红色警示信息


if __name__ == "__main__":  # 直接命令行运行入口
    parser = argparse.ArgumentParser()  # 参数解析器
    parser.add_argument("--question",  default="血压计袖带应该怎么绑？测出来的值才准？")  # 默认演示问题
    parser.add_argument("--max_steps", type=int, default=10)  # 最大步数默认 10
    args = parser.parse_args()  # 解析命令行参数
    run_and_print(args.question, args.max_steps)  # 执行并打印

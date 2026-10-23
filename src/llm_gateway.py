"""
LLM 网关 —— 统一接入层（P4 平台升级）

收敛 react_manual / react_function_calling 此前各自直连的 OpenAI 兼容客户端，
两个 ReAct 实现统一经 chat() 调 LLM，横切能力集中到网关一处：

  1. 主链路：DashScope 兼容端点（沿用既有变量命名：DASHSCOPE_API_KEY + AGENT_MODEL）
  2. 备链路（可选）：BP_LLM_FALLBACK_BASE_URL / BP_LLM_FALLBACK_API_KEY /
     BP_LLM_FALLBACK_MODEL —— 主链路重试耗尽且备链路已配置时，自动切换备用客户端再试一轮
  3. 显式重试：BP_LLM_MAX_RETRIES 语义不变（默认 2），重试挪到网关层统一实现，
     SDK 客户端构造时 max_retries=0，避免两层重试相乘放大调用次数
  4. 计时埋点：每次尝试独立包 llm_call_timer，llm_calls_total{status=ok|error}
     与 llm_call_duration_seconds 在网关内自动统计（调用方不再手写 with 块）
  5. 延迟初始化：单例客户端首次真实调用才构造（沿用"占位 Key 延迟报错"策略，
     无有效 Key 不在导入/构建时崩溃，错误推迟到 chat() 抛出）

测试注入约定：monkeypatch.setattr(llm_gateway, "_client", fake) 可整体替换主链路
客户端（Fake 需提供 .chat.completions.create(**kwargs) 接口，与 OpenAI SDK 同形）。
"""

import os      # 环境变量读取（Key/模型/重试/超时等全部配置）
import time    # 重试之间的指数退避等待
import logging  # 主备切换与重试告警日志

from observability import llm_call_timer  # LLM 调用统一埋点（每次尝试包一层）

logger = logging.getLogger(__name__)  # 本模块日志器

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 主链路端点（与原 react_manual 一致，代码内固定）

_client = None           # 主链路单例客户端（延迟初始化；测试可注入 Fake）
_fallback_client = None  # 备链路单例客户端（仅 BP_LLM_FALLBACK_* 配置时才会构造）


# ── 配置读取（每次调用时动态读环境变量，便于测试按用例覆盖） ──────────────────
def _max_retries() -> int:
    """显式重试次数（BP_LLM_MAX_RETRIES，默认 2；0=不重试只打一次）"""
    return max(0, int(os.getenv("BP_LLM_MAX_RETRIES", "2")))  # 负数防御性截为 0


def _retry_backoff() -> float:
    """重试基础退避秒数（BP_LLM_RETRY_BACKOFF，默认 0.5；按 2^n 指数增长，测试可设 0 消除等待）"""
    return max(0.0, float(os.getenv("BP_LLM_RETRY_BACKOFF", "0.5")))  # 负数防御性截为 0


def get_model() -> str:
    """主链路模型名（AGENT_MODEL，默认 qwen-max，沿用原 react_manual 命名）"""
    return os.getenv("AGENT_MODEL", "qwen-max")  # 环境变量可覆盖


def fallback_enabled() -> bool:
    """备链路是否启用：BP_LLM_FALLBACK_BASE_URL 与 BP_LLM_FALLBACK_API_KEY 同时配置才生效"""
    return bool(os.getenv("BP_LLM_FALLBACK_BASE_URL", "").strip()  # 端点与 Key 缺一不可
                and os.getenv("BP_LLM_FALLBACK_API_KEY", "").strip())


def get_fallback_model() -> str:
    """备链路模型名（BP_LLM_FALLBACK_MODEL，留空沿用主链路模型）"""
    return os.getenv("BP_LLM_FALLBACK_MODEL", "").strip() or get_model()  # 未配置时回落主链路模型名


# ── 客户端构造（延迟单例：占位 Key 延迟报错策略） ─────────────────────────────
def get_client():
    """主链路单例客户端：首次调用才构造（导入本模块零副作用，无 Key 也不崩）。"""
    global _client  # 模块级单例缓存
    if _client is None:  # 未初始化才构造
        from openai import OpenAI  # 延迟导入：纯单测导入本模块不连带拉起 OpenAI SDK
        _client = OpenAI(  # DashScope 的 OpenAI 兼容端点
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 主链路 Key（沿用原变量名，缺失时由 SDK 在调用时报错）
            base_url=DASHSCOPE_BASE_URL,  # 主链路端点
            max_retries=0,  # SDK 层重试关闭：统一由网关显式重试，避免两层相乘
        )
    return _client  # 返回单例


def get_fallback_client():
    """备链路单例客户端：仅 fallback_enabled() 为真时才会被构造。"""
    global _fallback_client  # 模块级单例缓存
    if _fallback_client is None:  # 未初始化才构造
        from openai import OpenAI  # 延迟导入：与 get_client 同策略
        _fallback_client = OpenAI(  # 备用供应商的 OpenAI 兼容端点
            api_key=os.getenv("BP_LLM_FALLBACK_API_KEY"),  # 备链路 Key
            base_url=os.getenv("BP_LLM_FALLBACK_BASE_URL"),  # 备链路端点
            max_retries=0,  # SDK 层重试关闭：与主链路口径一致
        )
    return _fallback_client  # 返回单例


# ── 单链路重试轮：1 + BP_LLM_MAX_RETRIES 次尝试，每次独立埋点 ─────────────────
def _call_with_retry(client, model: str, payload: dict):
    """在指定客户端上执行 create 调用并显式重试（每次尝试独立计入 llm_call_timer）。

    全部尝试失败时抛出最后一次异常（交由上层 chat 决定是否切备链路或上抛）。
    """
    attempts = 1 + _max_retries()  # 总尝试次数 = 首次 + 重试次数
    for i in range(attempts):  # 逐次尝试
        try:  # 单次尝试
            with llm_call_timer():  # 埋点：成功计 ok / 异常计 error 并原样上抛
                return client.chat.completions.create(model=model, **payload)  # 真正的 LLM 调用
        except Exception as e:  # 本尝试失败（网络抖动/超时/限流等）
            if i == attempts - 1:  # 已是最后一次尝试：不再等待，向上抛出
                raise  # 原样上抛（不吞错，主备切换决策在 chat 层）
            backoff = _retry_backoff() * (2 ** i)  # 指数退避：base * 2^i（0.5s/1s/2s...，测试可置 0）
            logger.warning("LLM 调用失败（第 %d/%d 次尝试，%.1fs 后重试）: %s",
                           i + 1, attempts, backoff, e)  # 重试告警（SDK 异常不含密钥）
            if backoff > 0:  # 退避为零则立即重试（测试路径）
                time.sleep(backoff)  # 真实等待，防打爆对端


# ── 统一调用入口 ───────────────────────────────────────────────────────────────
def chat(messages: list, tools: list | None = None, **kwargs):
    """LLM 统一调用入口：埋点 + 显式重试 + 主备切换全部在网关内完成。

    参数：
      messages : OpenAI 对话消息列表（system/user/assistant/tool）
      tools    : 工具 Schema 列表（FC 协议；None 表示不启用工具调用，不传给 SDK）
      **kwargs : 透传给 create 的其他参数（temperature/stop/tool_choice/timeout 等）

    返回：SDK 原生响应对象（choices[0].message 等），与直连客户端调用完全同形。
    主备链路全部重试耗尽时抛出最后一次异常（调用方沿用既有 try/except 降级逻辑）。
    """
    payload = dict(kwargs)  # 复制一份，主备两轮复用同一入参
    payload.pop("model", None)  # 防御：模型名由网关按链路统一决定，忽略调用方误传
    payload["messages"] = messages  # 消息列表（必传）
    payload.setdefault("timeout", float(os.getenv("BP_LLM_TIMEOUT", "60")))  # 默认超时（原调用点语义上移至此）
    if tools is not None:  # FC 场景才携带 tools（部分兼容端点不接受 tools=None）
        payload["tools"] = tools  # 工具 Schema 透传
    try:  # —— 主链路重试轮 ——
        return _call_with_retry(get_client(), get_model(), payload)  # 主链路：DashScope + AGENT_MODEL
    except Exception as e:  # 主链路重试耗尽
        if not fallback_enabled():  # 未配置备链路：异常原样上抛（调用方降级）
            raise  # 保持与直连客户端一致的错误语义
        logger.warning("主链路重试耗尽（%s），切换备链路: %s",  # 主备切换留痕（供运维排查切换频次）
                       str(e)[:200], os.getenv("BP_LLM_FALLBACK_BASE_URL"))  # 错误摘要截断防刷屏
        return _call_with_retry(get_fallback_client(), get_fallback_model(), payload)  # 备链路再跑一轮重试

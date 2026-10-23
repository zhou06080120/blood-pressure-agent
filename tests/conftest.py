"""
blood_pressure_agent 测试全局配置

设计要点（离线可复现，本地与 CI 行为一致）：
  1. 占位 API Key：react_manual / react_function_calling 在模块级构造 OpenAI
     客户端，无 Key 会在 import 时崩溃 —— 这里预置占位值，真实调用由各测试
     用 Fake 客户端 monkeypatch 替换，永不触网。
  2. 强制模型降级：BP_MODEL_PATH / BP_RERANK_MODEL 指向不存在的路径，
     使 BGE embedding 与 CrossEncoder 重排均加载失败 → 工具链走确定性降级
     路径（关键词匹配 / TF 向量 / 词面重排），本地真实模型不干扰测试。
  3. autouse fixture：每个用例前后清空两级语义缓存，避免用例间污染。
"""

import os
import time

# ── 环境变量必须在任何 src 模块导入之前设置（pytest 加载本文件先于测试模块）──
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-placeholder")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("AGENT_MODEL", "qwen-max-test")
os.environ.setdefault("RATE_LIMIT_RPM", "0")  # 测试默认关闭限流（serve 导入时读此变量决定是否注册中间件）；限流专项测试自建 app
os.environ["BP_EMBED_PROVIDER"] = "local"
# 指向确定不存在的模型路径 → _load_st_model() / CrossEncoder 加载失败 → 走降级
_NO_MODEL = os.path.join(os.path.dirname(__file__), "_definitely_no_model_here")
os.environ["BP_MODEL_PATH"] = _NO_MODEL
os.environ["BP_RERANK_MODEL"] = _NO_MODEL

import pytest  # noqa: E402  (环境变量必须先行，故延迟导入)


# ── Fake LLM 客户端：按脚本回放，绝不触网 ─────────────────────────────────────

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeLLMClient:
    """按预设脚本依次返回 LLM 输出的假客户端。

    usage:
        client = FakeLLMClient(["Thought: ...\\nFinal Answer: ..."])
        monkeypatch.setattr(react_manual, "client", client)
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # 记录每次 create 的入参，供断言调用次数/内容
        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if not outer.script:
                    raise AssertionError("LLM 调用次数超出 FakeClient 脚本预设")
                return _FakeResponse(outer.script.pop(0))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.fixture()
def fake_llm():
    """返回 FakeLLMClient 工厂，便于在用例中按需构造脚本"""
    return FakeLLMClient


# ── FC 版假客户端：支持 tool_calls / finish_reason 的完整协议模拟 ──────────────

class FCToolCall:
    """OpenAI tool_call 协议桩：id + function.name + function.arguments(JSON 字符串)"""

    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


class FCMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls  # None 表示无工具调用（stop 路径）


class FCResponse:
    def __init__(self, message, finish_reason="stop"):
        self.choices = [type("Choice", (), {"message": message,
                                            "finish_reason": finish_reason})()]


class FCClient:
    """FC 版假客户端：脚本元素为 FCResponse 实例，依次回放"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if not outer.script:
                    raise AssertionError("LLM 调用次数超出 FCClient 脚本预设")
                item = outer.script.pop(0)
                if isinstance(item, str):  # str 简写 = 直接给出最终答案
                    item = FCResponse(FCMessage(content=item), finish_reason="stop")
                return item

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def fc_final(text: str) -> FCResponse:
    """构造直接给出最终答案的 FC 响应（finish_reason=stop）"""
    return FCResponse(FCMessage(content=text), finish_reason="stop")


def fc_tool_calls(*calls: FCToolCall) -> FCResponse:
    """构造请求工具调用的 FC 响应（finish_reason=tool_calls）"""
    return FCResponse(FCMessage(tool_calls=list(calls)), finish_reason="tool_calls")


@pytest.fixture()
def fc_llm():
    """返回 (FCClient 工厂, FCToolCall 工厂) 元组"""
    return FCClient, FCToolCall


# ── 缓存隔离：两级语义缓存全局单例，逐用例清空 ────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_semantic_cache():
    from semantic_cache import question_cache, tool_cache
    question_cache._entries.clear()
    question_cache._hits = question_cache._misses = question_cache._semantic_hits = 0
    tool_cache._entries.clear()
    tool_cache._hits = tool_cache._misses = 0
    yield
    question_cache._entries.clear()
    question_cache._misses = 0
    tool_cache._entries.clear()


# ── 会话管理器隔离：serve 与 agent 共享 conv_manager 单例，逐用例重置 ─────────

@pytest.fixture(autouse=True)
def _clean_conversations():
    from conversation import conv_manager
    conv_manager._sessions.clear()
    yield
    conv_manager._sessions.clear()


# ── 混合检索单例隔离：避免跨用例复用已构建索引 ────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_hybrid_retriever(monkeypatch):
    import tools
    monkeypatch.setattr(tools, "_hybrid_retriever", None, raising=False)
    import troubleshoot_graph
    monkeypatch.setattr(troubleshoot_graph, "_graph", None, raising=False)
    yield


# ── 时间推进工具：语义缓存 TTL 过期测试用 ─────────────────────────────────────

@pytest.fixture()
def time_travel(monkeypatch):
    """返回偏移函数：将 time.time 人为前移 N 秒（只影响 semantic_cache 模块）"""
    def _advance(seconds: float):
        real_time = time.time
        monkeypatch.setattr("semantic_cache.time.time", lambda: real_time() + seconds)
    return _advance

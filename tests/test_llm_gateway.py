"""llm_gateway 统一接入层测试：延迟单例、显式重试、主备切换、计时埋点

设计要点：
  - Fake 客户端直接注入 llm_gateway._client / _fallback_client 单例（与业务测试同款手法）
  - BP_LLM_MAX_RETRIES / BP_LLM_RETRY_BACKOFF 按用例显式设置（conftest 默认重试关闭）
  - 指标断言直接读 observability.LLM_CALLS 计数器（全进程单例，"前值→动作→增量"口径）
"""

import pytest

import llm_gateway
from observability import LLM_CALLS


# ═══ Fake 客户端：脚本回放（异常对象=抛出，其余=返回） ═════════════════════════

class _Resp:
    """最小 OpenAI 响应形桩：choices[0].message.content"""

    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class _ScriptedClient:
    """按脚本回放的假客户端：脚本元素为 Exception 则抛出，否则作为响应返回"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # 每次 create 的入参记录
        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if not outer.script:  # 脚本耗尽：视为测试编程错误
                    raise AssertionError("Fake 客户端脚本耗尽")
                item = outer.script.pop(0)
                if isinstance(item, Exception):  # 异常脚本项：模拟调用失败
                    raise item
                return item  # 正常响应

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _ok_count() -> float:
    """llm_calls_total{status=ok} 当前累计值"""
    return LLM_CALLS.labels("ok")._value.get()


def _err_count() -> float:
    """llm_calls_total{status=error} 当前累计值"""
    return LLM_CALLS.labels("error")._value.get()


# ═══ 客户端构造：延迟单例 ══════════════════════════════════════════════════════

class TestClientSingleton:
    def test_get_client_lazy_singleton(self, monkeypatch):
        monkeypatch.setattr(llm_gateway, "_client", None)  # 从未初始化状态起步（conftest 已设占位 Key，构造不触网）
        c1 = llm_gateway.get_client()
        c2 = llm_gateway.get_client()
        assert c1 is c2  # 单例复用：第二次直接返回缓存实例

    def test_get_fallback_client_lazy_singleton(self, monkeypatch):
        monkeypatch.setattr(llm_gateway, "_fallback_client", None)
        monkeypatch.setenv("BP_LLM_FALLBACK_BASE_URL", "https://backup.example.com/v1")
        monkeypatch.setenv("BP_LLM_FALLBACK_API_KEY", "sk-backup")
        f1 = llm_gateway.get_fallback_client()
        f2 = llm_gateway.get_fallback_client()
        assert f1 is f2  # 备链路同样单例复用


# ═══ 统一入口：payload 组装 ════════════════════════════════════════════════════

class TestChatPayload:
    def test_model_and_default_timeout_injected(self, monkeypatch):
        client = _ScriptedClient([_Resp("ok")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        llm_gateway.chat([{"role": "user", "content": "q"}], temperature=0)
        kw = client.calls[0]
        assert kw["model"] == "qwen-max-test"  # 网关统一注入模型名（AGENT_MODEL）
        assert kw["timeout"] == 60.0  # 默认超时自动补齐
        assert "tools" not in kw  # 未传 tools 不携带该键（兼容不接受 tools=None 的端点）
        assert kw["messages"] == [{"role": "user", "content": "q"}]  # 消息透传

    def test_tools_forwarded_when_given(self, monkeypatch):
        client = _ScriptedClient([_Resp("ok"), _Resp("ok")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        schema = [{"type": "function", "function": {"name": "faq_search"}}]
        llm_gateway.chat([{"role": "user", "content": "q"}])  # 第一次不带 tools
        llm_gateway.chat([{"role": "user", "content": "q"}], tools=schema)  # 第二次带 tools
        assert "tools" not in client.calls[0]  # 不带：无该键
        assert client.calls[1]["tools"] == schema  # 带上：原样透传

    def test_success_single_call_counts_ok(self, monkeypatch):
        client = _ScriptedClient([_Resp("ok")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "2")  # 即使允许重试
        before = _ok_count()
        resp = llm_gateway.chat([{"role": "user", "content": "q"}])
        assert resp.choices[0].message.content == "ok"  # 响应原样返回
        assert len(client.calls) == 1  # 成功不重试：恰好一次
        assert _ok_count() - before == 1  # llm_calls_total{ok} 恰好 +1


# ═══ 显式重试：BP_LLM_MAX_RETRIES 语义 ═════════════════════════════════════════

class TestRetry:
    def test_retry_then_success(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "2")  # 允许重试 2 次
        monkeypatch.setenv("BP_LLM_RETRY_BACKOFF", "0")  # 零退避：测试不等待
        client = _ScriptedClient([RuntimeError("boom"), RuntimeError("boom"), _Resp("ok")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        before_ok, before_err = _ok_count(), _err_count()
        resp = llm_gateway.chat([{"role": "user", "content": "q"}])
        assert resp.choices[0].message.content == "ok"  # 第三次尝试成功
        assert len(client.calls) == 3  # 总尝试 = 1 首次 + 2 重试
        assert _err_count() - before_err == 2  # 两次失败各计一次 error
        assert _ok_count() - before_ok == 1  # 最终成功计一次 ok

    def test_retry_exhausted_raises_without_fallback(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "1")  # 允许重试 1 次
        monkeypatch.setenv("BP_LLM_RETRY_BACKOFF", "0")
        monkeypatch.delenv("BP_LLM_FALLBACK_BASE_URL", raising=False)  # 未配置备链路
        monkeypatch.delenv("BP_LLM_FALLBACK_API_KEY", raising=False)
        client = _ScriptedClient([RuntimeError("down"), RuntimeError("down")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        with pytest.raises(RuntimeError, match="down"):  # 重试耗尽：最后一次异常原样上抛
            llm_gateway.chat([{"role": "user", "content": "q"}])
        assert len(client.calls) == 2  # 总尝试 = 1 首次 + 1 重试

    def test_zero_retries_single_attempt(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "0")  # 0 = 不重试
        client = _ScriptedClient([RuntimeError("once")])
        monkeypatch.setattr(llm_gateway, "_client", client)
        with pytest.raises(RuntimeError, match="once"):
            llm_gateway.chat([{"role": "user", "content": "q"}])
        assert len(client.calls) == 1  # 恰好一次（conftest 全局默认即此口径）


# ═══ 主备切换 ══════════════════════════════════════════════════════════════════

class TestFallback:
    def test_fallback_takes_over_after_main_exhausted(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "1")
        monkeypatch.setenv("BP_LLM_RETRY_BACKOFF", "0")
        monkeypatch.setenv("BP_LLM_FALLBACK_BASE_URL", "https://backup.example.com/v1")
        monkeypatch.setenv("BP_LLM_FALLBACK_API_KEY", "sk-backup")
        monkeypatch.setenv("BP_LLM_FALLBACK_MODEL", "backup-model")
        main = _ScriptedClient([RuntimeError("main down"), RuntimeError("main down")])  # 主链路必炸
        backup = _ScriptedClient([_Resp("from-backup")])  # 备链路一次成功
        monkeypatch.setattr(llm_gateway, "_client", main)
        monkeypatch.setattr(llm_gateway, "_fallback_client", backup)
        resp = llm_gateway.chat([{"role": "user", "content": "q"}])
        assert resp.choices[0].message.content == "from-backup"  # 备链路结果返回
        assert len(main.calls) == 2  # 主链路先跑满一轮（1+1 次）
        assert len(backup.calls) == 1  # 备链路接手后一次成功
        assert backup.calls[0]["model"] == "backup-model"  # 备链路模型名生效
        assert backup.calls[0]["messages"] == main.calls[0]["messages"]  # 主备收到相同消息

    def test_fallback_requires_both_base_url_and_key(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_FALLBACK_BASE_URL", "https://backup.example.com/v1")
        monkeypatch.delenv("BP_LLM_FALLBACK_API_KEY", raising=False)  # 只配端点不配 Key
        assert llm_gateway.fallback_enabled() is False  # 不启用（端点与 Key 缺一不可）

    def test_fallback_model_defaults_to_main(self, monkeypatch):
        monkeypatch.delenv("BP_LLM_FALLBACK_MODEL", raising=False)  # 备链路模型未配置
        monkeypatch.setenv("AGENT_MODEL", "qwen-max-test")
        assert llm_gateway.get_fallback_model() == "qwen-max-test"  # 沿用主链路模型名

    def test_no_fallback_client_built_when_disabled(self, monkeypatch):
        monkeypatch.setenv("BP_LLM_MAX_RETRIES", "0")
        monkeypatch.delenv("BP_LLM_FALLBACK_BASE_URL", raising=False)
        monkeypatch.delenv("BP_LLM_FALLBACK_API_KEY", raising=False)
        monkeypatch.setattr(llm_gateway, "_client", _ScriptedClient([RuntimeError("down")]))
        monkeypatch.setattr(llm_gateway, "_fallback_client", None)  # 从未构造起步
        with pytest.raises(RuntimeError):
            llm_gateway.chat([{"role": "user", "content": "q"}])
        assert llm_gateway._fallback_client is None  # 备链路未启用：绝不构造备用客户端

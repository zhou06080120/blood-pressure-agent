"""conversation_store 会话存储测试：Memory 直查、Redis（fakeredis 注入）往返、回落逻辑

设计要点：
  - fakeredis 为纯 Python 的 dev 依赖：Redis 逻辑全程离线验证，无需真实 Redis
  - 回落逻辑用"构造必炸的桩类"模拟 Redis 不可用，验证 warning + Memory 回落
  - ConversationManager(store=...) 注入测试：管理器与存储解耦的端到端验证
"""

import logging
import time

import pytest

from conversation import ConversationManager
from conversation_store import (MemoryConversationStore, RedisConversationStore,
                                create_store)

fakeredis = pytest.importorskip("fakeredis")  # dev 依赖缺失时跳过本文件而非报错


# ═══ MemoryConversationStore：原行为搬迁验证 ════════════════════════════════════

class TestMemoryStore:
    def test_put_get_roundtrip(self):
        s = MemoryConversationStore()
        s.put("s1", {"messages": [{"role": "user", "content": "问", "timestamp": 1.0}],
                     "last_active": 1.0})
        rec = s.get("s1")
        assert rec["messages"][0]["content"] == "问"  # 消息原样读回
        assert rec["last_active"] == 1.0  # 活跃时间原样读回

    def test_get_missing_returns_none(self):
        assert MemoryConversationStore().get("no-such") is None  # 不存在返回 None

    def test_append_and_truncate(self):
        s = MemoryConversationStore()
        s.put("s1", {"messages": [], "last_active": 0.0})
        assert s.append("s1", "user", "m0") is True  # 追加成功
        s.append("s1", "user", "m1")
        s.append("s1", "user", "m2", max_history=2)  # 上限 2：截掉最旧的 m0
        rec = s.get("s1")
        assert [m["content"] for m in rec["messages"]] == ["m1", "m2"]  # 保留最近 2 条
        assert all(m["timestamp"] > 0 for m in rec["messages"])  # 自动盖时间戳
        assert rec["last_active"] > 0  # 活跃时间已刷新

    def test_append_missing_session_fails(self):
        assert MemoryConversationStore().append("nope", "user", "x") is False  # 会话不存在写入失败

    def test_clear_keeps_shell(self):
        s = MemoryConversationStore()
        s.put("s1", {"messages": [{"role": "user", "content": "q", "timestamp": 1.0}],
                     "last_active": 1.0})
        assert s.clear("s1") is True  # 清空成功
        rec = s.get("s1")
        assert rec["messages"] == []  # 消息已清空
        assert rec["last_active"] >= 1.0  # 会话壳保留且活跃时间刷新
        assert s.clear("nope") is False  # 不存在清空失败

    def test_delete_and_list(self):
        s = MemoryConversationStore()
        s.put("s1", {"messages": [], "last_active": 0.0})
        s.put("s2", {"messages": [], "last_active": 0.0})
        assert sorted(s.list_sessions()) == ["s1", "s2"]  # 列出全部会话 id
        assert s.delete("s1") is True  # 删除成功
        assert s.delete("s1") is False  # 重复删除失败
        assert s.list_sessions() == ["s2"]  # 仅剩 s2


# ═══ RedisConversationStore：fakeredis 注入往返 ════════════════════════════════

class TestRedisStore:
    def test_put_append_get_roundtrip(self):
        s = RedisConversationStore(client=fakeredis.FakeRedis())
        s.put("s1", {"messages": [], "last_active": 0.0})
        assert s.append("s1", "user", "问") is True  # 追加成功
        assert s.append("s1", "assistant", "答") is True
        rec = s.get("s1")
        assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]  # 消息顺序完整
        assert rec["messages"][0]["content"] == "问"  # 中文原样往返（JSON ensure_ascii=False）
        assert rec["last_active"] > 0  # 活跃时间刷新

    def test_append_truncate_on_redis(self):
        s = RedisConversationStore(client=fakeredis.FakeRedis())
        s.put("s1", {"messages": [], "last_active": 0.0})
        for i in range(4):
            s.append("s1", "user", f"m{i}", max_history=2)  # 每次追加都按上限截断
        assert [m["content"] for m in s.get("s1")["messages"]] == ["m2", "m3"]  # 只留最近 2 条

    def test_clear_and_delete(self):
        s = RedisConversationStore(client=fakeredis.FakeRedis())
        s.put("s1", {"messages": [{"role": "user", "content": "q", "timestamp": 1.0}],
                     "last_active": 1.0})
        assert s.clear("s1") is True  # 清空保留壳
        assert s.get("s1")["messages"] == []
        assert s.clear("nope") is False  # 不存在失败
        assert s.delete("s1") is True  # 彻底删除
        assert s.get("s1") is None  # 删除后读不到
        assert s.delete("s1") is False  # 重复删除失败

    def test_list_sessions_via_scan(self):
        s = RedisConversationStore(client=fakeredis.FakeRedis())
        s.put("a", {"messages": [], "last_active": 0.0})
        s.put("b", {"messages": [], "last_active": 0.0})
        s._r.set("other:key", "x")  # 命名空间外的键：不应被列出
        assert sorted(s.list_sessions()) == ["a", "b"]  # SCAN 只取本命名空间会话 id


# ═══ ConversationManager + 可插拔 Store：端到端 ════════════════════════════════

class TestManagerWithStore:
    def test_manager_with_memory_store(self):
        m = ConversationManager(max_history=2, store=MemoryConversationStore())
        sid = m.create_session()
        m.add_message(sid, "user", "q1")
        m.add_message(sid, "assistant", "a1")
        m.add_message(sid, "user", "q2")
        m.add_message(sid, "assistant", "a2")  # 超过 max_history=2：截断
        history = m.get_history(sid)
        assert len(history) == 2  # 只保留最近 2 条
        assert history[0]["content"] == "q2"  # 最旧的 q1/a1 被裁掉
        assert m.get_info(sid)["message_count"] == 2  # 元信息与历史一致

    def test_manager_with_redis_store(self):
        m = ConversationManager(store=RedisConversationStore(client=fakeredis.FakeRedis()))
        sid = m.create_session()
        m.add_message(sid, "user", "袖带怎么绑？")
        m.add_message(sid, "assistant", "下缘距肘窝2-3厘米。")
        assert len(m.get_history(sid)) == 2  # 往返完整
        assert m.get_info(sid)["messages"][0]["content"] == "袖带怎么绑？"  # 中文无损
        assert m.clear(sid) is True  # 清空
        assert m.get_history(sid) == []
        assert m.destroy(sid) is True  # 销毁
        assert m.get_info(sid) is None

    def test_manager_ttl_cleanup_on_redis_store(self):
        m = ConversationManager(session_ttl=0, store=RedisConversationStore(client=fakeredis.FakeRedis()))
        m.create_session()
        m.create_session()
        time.sleep(0.01)  # 确保 last_active < now（TTL=0 立即过期）
        assert m.cleanup_expired() == 2  # 过期清理走 list_sessions+delete（Redis 路径）


# ═══ create_store 工厂：选择与回落 ═════════════════════════════════════════════

class TestCreateStore:
    def test_default_memory_when_unset(self, monkeypatch):
        monkeypatch.delenv("BP_REDIS_URL", raising=False)  # 未配置
        assert isinstance(create_store(), MemoryConversationStore)  # 默认内存（零行为变化）

    def test_fallback_to_memory_with_warning_when_redis_down(self, monkeypatch, caplog):
        monkeypatch.setenv("BP_REDIS_URL", "redis://unreachable:9999/0")  # 配置了 Redis
        import conversation_store

        class _BoomStore:  # 构造必炸的桩：模拟 redis 缺包/连不上
            def __init__(self, *args, **kwargs):
                raise ConnectionError("redis down")

        monkeypatch.setattr(conversation_store, "RedisConversationStore", _BoomStore)
        with caplog.at_level(logging.WARNING, logger="conversation_store"):
            store = create_store()
        assert isinstance(store, MemoryConversationStore)  # 自动回落内存
        assert "回落" in caplog.text  # 启动时 warning 一次

    def test_redis_selected_when_configured_and_alive(self, monkeypatch):
        monkeypatch.setenv("BP_REDIS_URL", "redis://localhost:6379/0")

        class _AliveStore(MemoryConversationStore):  # 用内存实现冒充构造成功的 Redis（探活通过）
            def __init__(self, *args, **kwargs):  # 接受 create_store 传入的 url 等参数（构造即成功）
                super().__init__()

        import conversation_store
        monkeypatch.setattr(conversation_store, "RedisConversationStore", _AliveStore)
        assert isinstance(create_store(), _AliveStore)  # 配置且可用 → 选择 Redis 实现

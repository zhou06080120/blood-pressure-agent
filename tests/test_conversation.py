"""conversation 会话管理测试：创建/隔离/截断/清空/销毁/过期清理"""

import time

from conversation import ConversationManager


class TestLifecycle:
    def test_create_and_get_history(self):
        m = ConversationManager()
        sid = m.create_session()
        assert sid
        m.add_message(sid, "user", "袖带怎么绑？")
        m.add_message(sid, "assistant", "袖带下缘距肘窝2-3厘米")
        history = m.get_history(sid)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "袖带怎么绑？"}
        assert history[1]["role"] == "assistant"

    def test_history_isolated_between_sessions(self):
        m = ConversationManager()
        s1, s2 = m.create_session(), m.create_session()
        m.add_message(s1, "user", "问题1")
        assert m.get_history(s2) == []
        assert len(m.get_history(s1)) == 1

    def test_add_message_to_unknown_session_fails(self):
        m = ConversationManager()
        assert m.add_message("no-such-id", "user", "x") is False

    def test_get_history_unknown_returns_empty(self):
        m = ConversationManager()
        assert m.get_history("no-such-id") == []


class TestTruncation:
    def test_max_history_truncation(self):
        m = ConversationManager(max_history=4)
        sid = m.create_session()
        for i in range(6):
            m.add_message(sid, "user" if i % 2 == 0 else "assistant", f"msg{i}")
        history = m.get_history(sid)
        assert len(history) == 4          # 截断保留最近 4 条
        assert history[0]["content"] == "msg2"  # 最旧的 msg0/msg1 被裁掉


class TestClearDestroy:
    def test_clear_keeps_session_shell(self):
        m = ConversationManager()
        sid = m.create_session()
        m.add_message(sid, "user", "q")
        assert m.clear(sid) is True
        assert m.get_history(sid) == []
        assert m.get_info(sid) is not None  # 会话仍存在

    def test_clear_unknown_fails(self):
        assert ConversationManager().clear("nope") is False

    def test_destroy_removes_session(self):
        m = ConversationManager()
        sid = m.create_session()
        assert m.destroy(sid) is True
        assert m.get_info(sid) is None
        assert m.destroy(sid) is False


class TestExpiry:
    def test_cleanup_expired(self):
        m = ConversationManager(session_ttl=0)  # 立即过期
        m.create_session()
        m.create_session()
        time.sleep(0.01)  # 确保 last_active < now
        cleaned = m.cleanup_expired()
        assert cleaned == 2

    def test_active_session_not_cleaned(self):
        m = ConversationManager(session_ttl=3600)
        sid = m.create_session()
        assert m.cleanup_expired() == 0
        assert m.get_info(sid) is not None


class TestInfo:
    def test_get_info_fields(self):
        m = ConversationManager()
        sid = m.create_session()
        m.add_message(sid, "user", "x" * 300)  # 超长内容
        info = m.get_info(sid)
        assert info["session_id"] == sid
        assert info["message_count"] == 1
        assert len(info["messages"][0]["content"]) == 100  # 预览截断 100 字

    def test_list_sessions(self):
        m = ConversationManager()
        m.create_session()
        m.create_session()
        sessions = m.list_sessions()
        assert len(sessions) == 2
        assert {"session_id", "message_count", "last_active", "idle_seconds"} <= set(sessions[0])

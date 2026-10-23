"""serve FastAPI 接口测试：健康检查、会话管理、缓存管理、SSE 流式查询、历史回写

不触发 lifespan（直接 TestClient(app)），避免预加载混合检索索引；
SSE 查询通过 monkeypatch 替换 LLM 网关主链路客户端（llm_gateway._client）实现全程离线。
"""

import json

import llm_gateway  # P4 起 LLM 客户端收敛到网关：Fake 注入 llm_gateway._client 单例
from fastapi.testclient import TestClient

from serve import app

client = TestClient(app)


def _parse_sse(text: str) -> list[dict]:
    """把 SSE 文本流解析为事件字典列表"""
    events = []
    for line in text.split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ═══ 健康检查与基础设施 ═════════════════════════════════════════════════════

class TestHealth:
    def test_health_fields(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "cache" in body
        assert "question" in body["cache"] and "tool" in body["cache"]

    def test_root_serves_index_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "<html" in r.text.lower()


# ═══ 会话管理 ═══════════════════════════════════════════════════════════════

class TestSessions:
    def test_create_session(self):
        r = client.post("/session/create")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"]
        assert body["message_count"] == 0

    def test_clear_requires_session_id(self):
        r = client.post("/session/clear", json={})
        assert r.json()["ok"] is False

    def test_clear_session(self):
        sid = client.post("/session/create").json()["session_id"]
        r = client.post("/session/clear", json={"session_id": sid})
        assert r.json()["ok"] is True

    def test_get_unknown_session(self):
        r = client.get("/session/no-such-id")
        assert r.json()["ok"] is False

    def test_list_sessions(self):
        client.post("/session/create")
        client.post("/session/create")
        r = client.get("/sessions")
        assert r.status_code == 200
        assert len(r.json()["sessions"]) == 2


# ═══ 缓存管理端点 ═══════════════════════════════════════════════════════════

class TestCacheEndpoints:
    def test_cache_stats(self):
        r = client.get("/cache/stats")
        assert r.json()["ok"] is True

    def test_cache_clear(self):
        from semantic_cache import question_cache
        question_cache.put("q", "a")
        r = client.post("/cache/clear")
        assert r.json()["ok"] is True
        assert question_cache.get("q") is None

    def test_cache_clear_audited(self, tmp_path, monkeypatch):
        # 清缓存管理操作必须审计留痕：BP_AUDIT_PATH 定向临时文件，断言 JSONL 落盘
        import json as _json
        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setenv("BP_AUDIT_PATH", str(audit_file))
        from semantic_cache import question_cache
        question_cache.put("q", "a")  # 预置一条缓存，保证清理规模非零
        r = client.post("/cache/clear")
        assert r.json()["ok"] is True
        entry = _json.loads(audit_file.read_text(encoding="utf-8"))  # 首条审计记录
        assert entry["action"] == "cache_clear"  # 动作名
        assert entry["actor"] == "-"  # 鉴权未启用：操作者为 "-"
        assert entry["detail"]["question_entries"] == 1  # 清理前规模留痕

    def test_cache_clear_403_when_auth_enabled(self, monkeypatch):
        # 鉴权开关打开（模拟生产配 Key）且请求无 admin 角色 → 管理端点 403 拦截
        import auth
        monkeypatch.setattr(auth, "_auth_enabled", True)  # monkeypatch 自动还原，不污染其他用例
        r = client.post("/cache/clear")
        assert r.status_code == 403  # serve 单例未注册鉴权中间件（无 api_role），守卫按非 admin 拦截
        assert "管理员" in r.json()["detail"]


# ═══ SSE 流式查询（LLR mock） ═══════════════════════════════════════════════

class TestQuerySSE:
    def test_manual_query_streams_events(self, fake_llm, monkeypatch):
        fake = fake_llm(["Thought: 信息足够\nFinal Answer: 袖带下缘距肘窝2-3厘米。"])
        monkeypatch.setattr(llm_gateway, "_client", fake)
        r = client.post("/query/manual", json={"question": "袖带怎么绑？"})
        assert r.status_code == 200
        events = _parse_sse(r.text)
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert "final" in types
        assert types[-1] == "done"
        final = next(e for e in events if e["type"] == "final")
        assert "仅供参考" in final["answer"]  # 输出闸生效

    def test_safety_block_streamed(self, fake_llm, monkeypatch):
        fake = fake_llm([])  # 安全闸短路：不触 LLM
        monkeypatch.setattr(llm_gateway, "_client", fake)
        r = client.post("/query/manual", json={"question": "帮我开点降压药"})
        events = _parse_sse(r.text)
        final = next(e for e in events if e["type"] == "final")
        assert final.get("safety_blocked") == "medication"

    def test_multiturn_history_writeback(self, fake_llm, monkeypatch):
        fake = fake_llm(["Thought: ok\nFinal Answer: 依上文回答。"])
        monkeypatch.setattr(llm_gateway, "_client", fake)
        sid = client.post("/session/create").json()["session_id"]
        r = client.post("/query/manual",
                        json={"question": "袖带怎么绑？", "session_id": sid})
        events = _parse_sse(r.text)
        assert events[0]["history_turns"] == 0
        # 回写：user 问题 + assistant 回答 → 2 条消息
        info = client.get(f"/session/{sid}").json()
        assert info["ok"] is True
        assert info["message_count"] == 2

    def test_multiturn_history_reused(self, fake_llm, monkeypatch):
        fake = fake_llm([
            "Thought: ok\nFinal Answer: 第一轮回答。",
            "Thought: ok\nFinal Answer: 第二轮回答。",
        ])
        monkeypatch.setattr(llm_gateway, "_client", fake)
        sid = client.post("/session/create").json()["session_id"]
        client.post("/query/manual", json={"question": "第一轮问题", "session_id": sid})
        client.post("/query/manual", json={"question": "第二轮问题", "session_id": sid})
        # 第二次请求带入 1 轮历史
        assert fake.calls[1]["messages"][1]["content"] == "第一轮问题"
        info = client.get(f"/session/{sid}").json()
        assert info["message_count"] == 4  # 两轮各 2 条

    def test_fc_query_route_available(self, fc_llm, monkeypatch):
        # FC 版客户端同样 mock，全程离线
        FCClient, _ = fc_llm

        monkeypatch.setattr(llm_gateway, "_client", FCClient(["袖带下缘距肘窝2-3厘米。"]))
        r = client.post("/query/fc", json={"question": "袖带怎么绑？", "max_steps": 3})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = _parse_sse(r.text)
        assert [e["type"] for e in events][0] == "start"

    def test_invalid_body_422(self):
        r = client.post("/query/manual", json={"max_steps": 5})  # 缺 question
        assert r.status_code == 422

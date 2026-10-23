"""semantic_cache 测试：问题级精确/语义命中、TTL 过期、工具缓存、容量淘汰"""

import numpy as np

import semantic_cache
from semantic_cache import SemanticCache, ToolCache, _normalize


class TestNormalize:
    def test_strips_punct_and_space(self):
        assert _normalize("袖带 怎么绑？") == _normalize("袖带怎么绑")

    def test_lowercase(self):
        assert _normalize("HEM-7156") == "hem7156"


# ═══ 问题级缓存：精确命中 ═══════════════════════════════════════════════════

class TestExactHit:
    def test_exact_hit(self):
        c = SemanticCache()
        c.put("袖带怎么绑", "袖带下缘距肘窝2-3厘米")
        hit = c.get("袖带怎么绑")
        assert hit is not None
        answer, hit_type = hit
        assert hit_type == "exact"
        assert answer == "袖带下缘距肘窝2-3厘米"

    def test_normalized_exact_hit(self):
        c = SemanticCache()
        c.put("袖带 怎么绑？", "答案")
        answer, hit_type = c.get("袖带怎么绑")
        assert hit_type == "exact"
        assert answer == "答案"

    def test_miss_returns_none(self):
        c = SemanticCache()
        c.put("完全无关的问题", "答案")
        # 查询侧 _embed 返回 None（测试环境模型不可用）→ 只走精确匹配
        assert c.get("另一个问题") is None

    def test_stats_hit_rate(self):
        c = SemanticCache()
        c.put("q1", "a1")
        assert c.get("q1") is not None      # 命中
        assert c.get("q2") is None          # 未命中
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5


# ═══ 问题级缓存：语义命中（monkeypatch 嵌入向量） ═══════════════════════════

class TestSemanticHit:
    def test_semantic_hit_same_vector(self, monkeypatch):
        vec = np.array([1.0, 0.0, 0.0], dtype="float32")
        monkeypatch.setattr(semantic_cache, "_embed", lambda text: vec)
        c = SemanticCache(sim_threshold=0.92)
        c.put("血压计袖带怎么绑", "袖带答案")
        # 不同文本但嵌入向量相同（模拟语义相同）→ 语义命中
        answer, hit_type = c.get("袖带如何缠绕")
        assert hit_type == "semantic"
        assert answer == "袖带答案"
        assert c.stats()["semantic_hits"] == 1

    def test_semantic_miss_orthogonal_vector(self, monkeypatch):
        v1 = np.array([1.0, 0.0], dtype="float32")
        v2 = np.array([0.0, 1.0], dtype="float32")  # 正交 → 相似度 0
        state = {"vec": v1}
        monkeypatch.setattr(semantic_cache, "_embed", lambda text: state["vec"])
        c = SemanticCache(sim_threshold=0.92)
        c.put("问题A", "答案A")
        state["vec"] = v2  # 查询换成正交向量
        assert c.get("问题B") is None


# ═══ TTL 过期 ═══════════════════════════════════════════════════════════════

class TestTTL:
    def test_expired_exact(self, time_travel):
        c = SemanticCache(ttl=100)
        c.put("问题", "答案")
        time_travel(200)  # 时间前移 200 秒 → 条目过期
        assert c.get("问题") is None

    def test_tool_cache_expired(self, time_travel):
        tc = ToolCache(ttl=100)
        tc.put("faq_search", {"query": "袖带"}, "结果")
        time_travel(200)
        assert tc.get("faq_search", {"query": "袖带"}) is None


# ═══ 工具级缓存 ═════════════════════════════════════════════════════════════

class TestToolCache:
    def test_put_get_hit(self):
        tc = ToolCache()
        tc.put("product_lookup", {"model": "欧姆龙HEM-7156"}, "规格文本")
        assert tc.get("product_lookup", {"model": "欧姆龙HEM-7156"}) == "规格文本"
        assert tc.stats()["hits"] == 1

    def test_faq_search_cached_by_query_only(self):
        tc = ToolCache()
        tc.put("faq_search", {"query": "袖带", "top_k": 5}, "检索结果")
        # top_k 不同但 query 相同 → 命中同一缓存
        assert tc.get("faq_search", {"query": "袖带", "top_k": 3}) == "检索结果"

    def test_other_tools_keyed_by_full_args(self):
        tc = ToolCache()
        tc.put("calculator", {"expr": "1+1"}, "2")
        assert tc.get("calculator", {"expr": "2+2"}) is None  # 参数不同 → 未命中

    def test_miss_records(self):
        tc = ToolCache()
        assert tc.get("t", {"a": 1}) is None
        assert tc.stats()["misses"] == 1


# ═══ 容量淘汰 ═══════════════════════════════════════════════════════════════

class TestEviction:
    def test_evict_oldest_half_when_full(self, monkeypatch):
        monkeypatch.setattr(semantic_cache, "CACHE_MAX", 4)
        c = SemanticCache()
        for i in range(5):  # 超容量 1 条 → 淘汰最旧一半（2 条）
            c.put(f"问题{i}", f"答案{i}")
        assert len(c._entries) <= 4
        # 最旧的应被淘汰
        assert c.get("问题0") is None


# ═══ 汇总统计 ═══════════════════════════════════════════════════════════════

def test_cache_stats_shape():
    from semantic_cache import cache_stats
    stats = cache_stats()
    assert set(stats.keys()) == {"question", "tool"}
    assert "hit_rate" in stats["question"]
    assert "hit_rate" in stats["tool"]

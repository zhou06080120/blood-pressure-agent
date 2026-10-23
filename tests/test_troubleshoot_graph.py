"""troubleshoot_graph 决策图谱测试：BFS 多跳推演、检查项/就医节点、交互式 walk"""

from troubleshoot_graph import (
    GRAPH_EDGES,
    TroubleshootGraph,
    _find_edge_target,
    _mentioned,
    format_graph_result,
    graph_resolve,
)
from tools import TROUBLESHOOT_MAP


# ═══ 图谱推演 ═══════════════════════════════════════════════════════════════

class TestResolve:
    def test_exact_match_with_steps(self):
        result = graph_resolve("测不出值")
        assert result["matched"] is True
        assert len(result["steps"]) >= 4      # 4 个原因 + 深入展开步骤
        assert result["depth"] >= 1

    def test_multihop_links(self):
        result = graph_resolve("测不出值")
        linked = {s["linked"] for s in result["steps"] if s.get("linked")}
        assert "漏气" in linked               # 原因"袖带漏气" → 深入"漏气"症状
        assert "无法开机" in linked            # 原因"电池电量不足" → 供电排查

    def test_next_questions_generated(self):
        result = graph_resolve("测不出值")
        assert result["next_questions"]        # 多跳场景生成追问
        assert len(result["next_questions"]) <= 5

    def test_check_node_expansion(self):
        # "测量值偏高"的袖带/姿势原因 → 检查项节点展开
        result = graph_resolve("测量值偏高")
        checks = [s["check"] for s in result["steps"]]
        assert any("袖带绑法检查" in c for c in checks)

    def test_medical_node(self):
        result = graph_resolve("数值波动大")
        assert any("就医建议" in s["check"] for s in result["steps"])

    def test_unmatched_returns_supported(self):
        result = graph_resolve("屏幕碎裂")
        assert result["matched"] is False
        assert "测不出值" in result["supported"]
        assert result["steps"] == []

    def test_depth_limit(self):
        g = TroubleshootGraph(TROUBLESHOOT_MAP, max_depth=0)
        result = g.resolve("测不出值", max_depth=0)
        # depth_limit=0：根节点展开自身原因，但不深入下一跳症状
        assert all(s.get("linked") is None or s["linked"] for s in result["steps"])
        deep = g.resolve("测不出值")  # 不传参 → 用实例 max_depth=0
        assert not deep["next_questions"]

    def test_no_infinite_loop(self):
        # 蓝牙 ↔ APP 同步互相引用 → visited 防环必须生效
        result = graph_resolve("蓝牙连不上")
        assert result["matched"] is True
        # 不会死循环（能返回即为通过）


# ═══ 交互式推演 ═════════════════════════════════════════════════════════════

class TestWalk:
    def test_walk_with_feedback(self):
        g = TroubleshootGraph(TROUBLESHOOT_MAP)
        result = g.walk("测不出值", "袖带已经重新绑紧了，气管也插紧了")
        assert "next" in result and "resolved_check" in result

    def test_walk_without_feedback_returns_first_step(self):
        g = TroubleshootGraph(TROUBLESHOOT_MAP)
        result = g.walk("测不出值", "")
        assert result["resolved_check"] == 0
        assert result["next"] is not None


# ═══ 格式化与辅助 ═══════════════════════════════════════════════════════════

class TestHelpers:
    def test_format_matched(self):
        result = graph_resolve("测不出值")
        text = format_graph_result(result)
        assert "【故障排查决策路径：测不出值】" in text
        assert "步骤1" in text
        assert "深入排查" in text

    def test_format_unmatched(self):
        text = format_graph_result(graph_resolve("不存在症状"))
        assert "未找到故障" in text

    def test_find_edge_target(self):
        edges = GRAPH_EDGES["测不出值"]
        assert _find_edge_target(edges, "袖带未正确绑紧或漏气") == "漏气"
        assert _find_edge_target(edges, "电池电量不足") == "无法开机"
        assert _find_edge_target(edges, "毫无关联的原因") is None

    def test_mentioned(self):
        assert _mentioned("疑似原因：袖带未正确绑紧或漏气", "袖带已经绑紧") is True
        assert _mentioned("疑似原因：电池电量不足", "袖带已经绑紧") is False

"""tools 工具层测试：产品查询/沙箱计算器/测量指南/故障排查/图谱排查/降级检索/注册表一致性

测试环境强制 BGE 模型不可用（conftest BP_MODEL_PATH 指向不存在路径），
faq_search 走确定性关键词降级路径 —— 本地与 CI 行为一致。
"""

from tools import (
    TOOLS_MAP,
    TOOLS_SCHEMA,
    _build_hybrid_docs,
    _keyword_search,
    tool_calculator,
    tool_faq_search,
    tool_graph_troubleshoot,
    tool_measurement_guide,
    tool_product_lookup,
    tool_troubleshoot,
)


# ═══ 产品查询 ═══════════════════════════════════════════════════════════════

class TestProductLookup:
    def test_exact_match(self):
        result = tool_product_lookup("欧姆龙HEM-7156")
        assert "HEM-7156" in result
        assert "21次" in result          # 存储容量
        assert "欧姆龙（OMRON）" in result

    def test_fuzzy_match(self):
        result = tool_product_lookup("HEM-7156")
        assert "相似型号" in result
        assert "HEM-7156" in result

    def test_unknown_model_lists_supported(self):
        result = tool_product_lookup("不存在的型号XYZ")
        assert "未找到型号" in result
        assert "欧姆龙HEM-7156" in result  # 列出全部支持型号

    def test_strip_whitespace(self):
        assert tool_product_lookup("  欧姆龙HEM-7156  ") == tool_product_lookup("欧姆龙HEM-7156")


# ═══ 沙箱计算器 ═════════════════════════════════════════════════════════════

class TestCalculator:
    def test_arithmetic(self):
        assert tool_calculator("80 + (120-80)/3") == "93.333333"

    def test_math_functions(self):
        assert tool_calculator("sqrt(16)") == "4.0"

    def test_builtins_allowed(self):
        assert tool_calculator("round(3.14159, 2)") == "3.14"

    def test_invalid_expr_returns_error(self):
        result = tool_calculator("1/")
        assert result.startswith("计算出错")

    def test_division_by_zero(self):
        assert tool_calculator("1/0").startswith("计算出错")

    def test_sandbox_blocks_dunder_import(self):
        # __builtins__ 被清空 → 无法访问任意内建
        result = tool_calculator("__import__('os').getcwd()")
        assert result.startswith("计算出错")

    def test_unknown_name(self):
        assert tool_calculator("open('/etc/passwd')").startswith("计算出错")


# ═══ 测量指南 ═══════════════════════════════════════════════════════════════

class TestMeasurementGuide:
    def test_exact_topic(self):
        result = tool_measurement_guide("血压分级")
        assert "收缩压≥140" in result or "140" in result
        assert "135/85" in result  # 家庭自测阈值

    def test_fuzzy_topic(self):
        result = tool_measurement_guide("袖带")
        assert "匹配到以下主题" in result
        assert "袖带正确绑法指南" in result

    def test_unknown_topic_lists_all(self):
        result = tool_measurement_guide("如何烹饪")
        assert "未找到主题" in result
        assert "平均动脉压" in result  # 支持主题清单


# ═══ 故障排查（线性） ════════════════════════════════════════════════════════

class TestTroubleshoot:
    def test_exact_symptom(self):
        result = tool_troubleshoot("测不出值")
        assert "【故障：测不出值】" in result
        assert "可能原因" in result
        assert "解决方案" in result

    def test_fuzzy_symptom(self):
        result = tool_troubleshoot("蓝牙")
        assert "匹配到以下故障" in result

    def test_unknown_symptom(self):
        result = tool_troubleshoot("屏幕碎了")
        assert "未找到故障" in result


# ═══ 图谱排查（多跳决策路径） ════════════════════════════════════════════════

class TestGraphTroubleshoot:
    def test_multihop_decision_path(self):
        result = tool_graph_troubleshoot("测不出值")
        assert "决策路径" in result
        assert "深入排查" in result       # 多跳联动（漏气/无法开机）
        assert "步骤" in result

    def test_unknown_symptom_falls_back(self):
        result = tool_graph_troubleshoot("完全没见过的症状")
        assert "未找到故障" in result

    def test_medical_node_included(self):
        result = tool_graph_troubleshoot("数值波动大")
        assert "就医" in result           # 心律不齐 → 就医建议节点


# ═══ faq_search 降级路径（模型不可用 → 关键词匹配） ══════════════════════════

class TestFaqSearchDegraded:
    def test_keyword_search_hits_faq(self):
        result = tool_faq_search("袖带怎么绑")
        assert "FAQ-" in result           # 降级检索仍带来源编号
        assert "袖带" in result

    def test_keyword_search_garbage_query(self):
        result = tool_faq_search("今天天气怎么样")
        assert result == "未检索到相关内容"

    def test_keyword_search_top_k(self):
        assert "FAQ-" in _keyword_search("测量前准备注意事项", 3)


# ═══ 工具注册表与 Schema 一致性（FC 协议契约） ═══════════════════════════════

class TestRegistry:
    def test_tools_map_names(self):
        assert set(TOOLS_MAP.keys()) == {
            "faq_search", "product_lookup", "calculator",
            "measurement_guide", "troubleshoot", "graph_troubleshoot",
        }

    def test_schema_matches_map(self):
        schema_names = {s["function"]["name"] for s in TOOLS_SCHEMA}
        assert schema_names == set(TOOLS_MAP.keys())

    def test_schema_required_params(self):
        for s in TOOLS_SCHEMA:
            params = s["function"]["parameters"]
            assert set(params["required"]) <= set(params["properties"])
            assert s["function"]["description"]  # 每个工具必须有描述


# ═══ 混合检索语料构建 ═══════════════════════════════════════════════════════

class TestHybridDocs:
    def test_docs_from_three_sources(self):
        docs = _build_hybrid_docs()
        types = {d["type"] for d in docs}
        assert types == {"faq", "product", "troubleshoot"}
        assert len([d for d in docs if d["type"] == "faq"]) == 20
        assert len([d for d in docs if d["type"] == "product"]) == 6
        assert len([d for d in docs if d["type"] == "troubleshoot"]) == 9
        prod = next(d for d in docs if d["id"] == "PROD-HEM-7156")
        assert "21次" in prod["content"]

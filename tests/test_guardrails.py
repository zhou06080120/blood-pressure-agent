"""guardrails 测试：决策状态机（合法迁移/循环深度）+ 工具调用监视器（去重阻断/失败自纠错）"""

from guardrails import (
    AgentStateMachine,
    ToolCallMonitor,
    _is_error,
    make_guards,
)


# ═══ 决策状态机 ═════════════════════════════════════════════════════════════

class TestStateMachine:
    def test_legal_transition_init_to_planning(self):
        sm = AgentStateMachine()
        assert sm.state == sm.INIT
        assert sm.transition(sm.PLANNING) is True
        assert sm.state == sm.PLANNING

    def test_illegal_transition_rejected(self):
        sm = AgentStateMachine()
        # INIT 不允许直接跳 FINAL / EXECUTING
        assert sm.transition(sm.FINAL) is False
        assert sm.state == sm.INIT  # 状态保持不变

    def test_full_react_cycle(self):
        sm = AgentStateMachine()
        for target in (sm.PLANNING, sm.EXECUTING, sm.OBSERVING, sm.PLANNING, sm.FINAL):
            assert sm.transition(target) is True
        assert sm.state == sm.FINAL

    def test_depth_counts_executing_only(self):
        sm = AgentStateMachine()
        sm.transition(sm.PLANNING)
        sm.transition(sm.EXECUTING)          # depth 1
        sm.transition(sm.OBSERVING)
        sm.transition(sm.PLANNING)
        sm.transition(sm.EXECUTING)          # depth 2
        assert sm.depth == 2
        assert sm.loop_exhausted is False

    def test_loop_exhausted(self):
        sm = AgentStateMachine(max_depth=1)
        sm.transition(sm.PLANNING)
        sm.transition(sm.EXECUTING)
        assert sm.loop_exhausted is True

    def test_recovering_branch(self):
        sm = AgentStateMachine()
        sm.transition(sm.PLANNING)
        sm.transition(sm.EXECUTING)
        assert sm.transition(sm.RECOVERING) is True
        assert sm.transition(sm.PLANNING) is True  # 恢复后可重新规划

    def test_final_is_terminal(self):
        sm = AgentStateMachine()
        sm.transition(sm.PLANNING)
        sm.transition(sm.FINAL)
        assert sm.transition(sm.PLANNING) is False  # 终态不可迁出

    def test_snapshot_fields(self):
        sm = AgentStateMachine(max_depth=5)
        sm.transition(sm.PLANNING)
        snap = sm.snapshot()
        assert snap == {"state": sm.PLANNING, "depth": 0,
                        "max_depth": 5, "loop_exhausted": False}


# ═══ 工具调用监视器 ═════════════════════════════════════════════════════════

class TestToolCallMonitor:
    def test_first_calls_allowed(self):
        m = ToolCallMonitor(max_same_call=2)
        for _ in range(2):
            blocked, _ = m.check_before("calculator", {"expr": "1+1"})
            assert blocked is False

    def test_third_identical_call_blocked(self):
        m = ToolCallMonitor(max_same_call=2)
        args = {"expr": "1+1"}
        m.check_before("calculator", args)
        m.check_before("calculator", args)
        m.record("calculator", args, "2")
        m.record("calculator", args, "2")
        blocked, hint = m.check_before("calculator", args)
        assert blocked is True
        assert "勿重复调用" in hint or "不要重复" in hint
        assert m.blocked_duplicates == 1

    def test_key_stable_across_arg_order(self):
        k1 = ToolCallMonitor._key("faq_search", {"query": "袖带", "top_k": 5})
        k2 = ToolCallMonitor._key("faq_search", {"top_k": 5, "query": "袖带"})
        assert k1 == k2  # sort_keys 保证参数顺序无关

    def test_key_differs_by_tool(self):
        k1 = ToolCallMonitor._key("faq_search", {"query": "x"})
        k2 = ToolCallMonitor._key("troubleshoot", {"query": "x"})
        assert k1 != k2

    def test_same_tool_different_args_not_blocked(self):
        m = ToolCallMonitor(max_same_call=2)
        for expr in ("1+1", "2+2", "3+3"):
            blocked, _ = m.check_before("calculator", {"expr": expr})
            assert blocked is False  # 参数不同不算重复

    def test_soft_hint_after_many_same_tool_calls(self):
        m = ToolCallMonitor(max_same_call=2)
        for i in range(5):
            blocked, hint = m.check_before("faq_search", {"query": f"问题{i}"})
            m.record("faq_search", {"query": f"问题{i}"}, "正常结果")  # record 登记后才有调用历史
        assert blocked is False
        assert "切换策略" in hint  # 软提示引导换工具

    def test_consecutive_error_tracking(self):
        m = ToolCallMonitor(max_consec_errors=2)
        m.record("t1", {}, "工具参数错误: xxx")
        assert m.consec_errors == 1
        m.record("t2", {}, "未知工具 't2'")
        assert m.consec_errors == 2
        hint = m.recovery_hint()
        assert hint is not None
        assert "连续 2 次" in hint
        assert m.recoveries == 1
        assert m.consec_errors == 0  # 提示后复位

    def test_success_resets_error_counter(self):
        m = ToolCallMonitor(max_consec_errors=2)
        m.record("t", {}, "计算出错: x")
        m.record("t", {"a": 1}, "正常结果")  # 成功 → 清零
        assert m.consec_errors == 0
        assert m.recovery_hint() is None

    def test_stats(self):
        m = ToolCallMonitor()
        m.record("a", {}, "ok")
        m.stats()
        assert m.stats()["total_calls"] == 1


# ═══ 失败信号判定 + 工厂 ═══════════════════════════════════════════════════

class TestHelpers:
    def test_is_error_signals(self):
        assert _is_error("工具参数错误: missing kw") is True
        assert _is_error("未知工具 'x'") is True
        assert _is_error("计算出错: div by zero") is True
        assert _is_error("袖带下缘距肘窝2-3厘米") is False

    def test_make_guards_factory(self):
        sm, monitor = make_guards()
        assert isinstance(sm, AgentStateMachine)
        assert isinstance(monitor, ToolCallMonitor)

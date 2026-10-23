"""parallel_executor 测试：保序并发、单调用同步路径、异常结构化兜底"""

import time

from parallel_executor import _run_one, run_parallel


def _sleep_tool(seconds):
    time.sleep(seconds)
    return f"slept {seconds}"


def _boom(**kwargs):
    raise RuntimeError("工具内部炸了")


def _bad_signature(unexpected_kw=None):
    return "never"


class TestRunParallel:
    def test_empty(self):
        assert run_parallel([]) == []

    def test_single_call_synchronous(self):
        results = run_parallel([{"name": "t", "fn": lambda: "ok", "args": {}}])
        assert len(results) == 1
        assert results[0]["observation"] == "ok"
        assert results[0]["error"] is None

    def test_parallel_preserves_order(self):
        # 第一个任务最慢：若未按原顺序回填会乱序
        calls = [
            {"name": "slow", "fn": _sleep_tool, "args": {"seconds": 0.15}},
            {"name": "fast", "fn": _sleep_tool, "args": {"seconds": 0.01}},
            {"name": "mid", "fn": _sleep_tool, "args": {"seconds": 0.05}},
        ]
        results = run_parallel(calls)
        assert [r["name"] for r in results] == ["slow", "fast", "mid"]

    def test_parallel_faster_than_serial(self):
        calls = [{"name": f"t{i}", "fn": _sleep_tool, "args": {"seconds": 0.1}}
                 for i in range(3)]
        t0 = time.perf_counter()
        run_parallel(calls)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.28  # 串行至少 0.3s，并发应显著更快（4 worker）

    def test_exception_becomes_structured_error(self):
        results = run_parallel([{"name": "boom", "fn": _boom, "args": {}}])
        r = results[0]
        assert r["error"] is not None
        assert "工具执行异常" in r["observation"]

    def test_typeerror_reports_bad_args(self):
        r = _run_one({"name": "bad", "fn": _bad_signature, "args": {"wrong": 1}})
        assert "工具参数错误" in r["observation"]

    def test_elapsed_ms_recorded(self):
        r = _run_one({"name": "t", "fn": lambda: "ok", "args": {}})
        assert r["elapsed_ms"] >= 0
        assert r["error"] is None

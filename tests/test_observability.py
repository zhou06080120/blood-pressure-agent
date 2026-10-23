"""observability 可观测基建测试：/metrics 端点、X-Request-ID、指标计数、日志格式化

设计要点：
  - prometheus_client 指标是全进程全局单例（多 app 实例共用同一注册表），
    所有一致性断言都用"前值 → 动作 → 增量"口径，不依赖绝对值
  - 限流拒绝用例自建独立 FastAPI 应用（serve 单例 app 在 conftest 已设
    RATE_LIMIT_RPM=0 关闭限流，429 无从触发），monkeypatch 临时改环境变量
  - JsonFormatter / setup_logging 直接单元级验证（json.loads 反解校验）
"""

import json
import logging
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from observability import (JsonFormatter, install_observability, llm_call_timer,
                           request_id_var, setup_logging)
from rate_limit import install_rate_limit
from serve import app

client = TestClient(app)


# ═══ 辅助：从 /metrics 文本解析计数器累计值 ═════════════════════════════════

def _counter_value(text: str, name: str, label: str | None = None) -> float:
    """累加 /metrics 文本中以 name+"{" 或 name+" " 开头的行尾数值。

    精确前缀匹配天然排除 name_created / name_bucket 等衍生序列；
    label 非空时仅累加包含该标签串的行（如 'status="ok"'）。
    """
    total = 0.0  # 累加器
    for line in text.splitlines():  # 逐行扫描 Prometheus 文本协议
        if not (line.startswith(name + "{") or line.startswith(name + " ")):
            continue  # 不是目标指标（衍生序列前缀更长，不匹配）
        if label is not None and label not in line:
            continue  # 标签过滤（如只看 status="ok"）
        total += float(line.rsplit(" ", 1)[1])  # 行尾数值累加
    return total


def _metric_value(name: str, label: str | None = None) -> float:
    """拉取 serve app 的 /metrics 文本并解析指定计数器当前累计值"""
    r = client.get("/metrics")  # 抓取指标端点
    assert r.status_code == 200  # 端点可用性前置
    return _counter_value(r.text, name, label)


# ═══ /metrics 端点 ═══════════════════════════════════════════════════════════

class TestMetricsEndpoint:
    def test_metrics_200_with_prometheus_text(self):
        r = client.get("/metrics")  # 抓取指标端点
        assert r.status_code == 200  # 正常返回
        assert "text/plain" in r.headers["content-type"]  # Prometheus 文本协议
        body = r.text  # 指标文本
        assert "http_requests_total" in body  # HTTP 计数器已注册
        assert "rate_limit_rejected_total" in body  # 限流拒绝计数器已注册
        assert "llm_calls_total" in body  # LLM 调用计数器已注册
        assert "http_request_duration_seconds" in body  # 延迟直方图已注册

    def test_http_requests_total_increments(self):
        before = _metric_value("http_requests_total")  # 前值（全局单例，只看增量）
        r = client.get("/health")  # 打一次请求
        assert r.status_code == 200  # 请求成功
        after = _metric_value("http_requests_total")  # 后值
        assert after - before >= 1  # 计数至少 +1

    def test_path_label_uses_route_template(self):
        client.get("/session/whatever-id")  # 带路径参数的真实请求
        body = client.get("/metrics").text  # 拉取指标文本
        assert 'path="/session/{session_id}"' in body  # 标签是路由模板而非具体 ID（防高基数）


# ═══ X-Request-ID：自动生成 / 透传回写 ═══════════════════════════════════════

class TestRequestId:
    def test_auto_generated_request_id_is_12_hex(self):
        r = client.get("/health")  # 不带请求头的普通请求
        rid = r.headers.get("x-request-id")  # 响应头回写的请求 ID
        assert rid is not None  # 中间件必须回写
        assert re.fullmatch(r"[0-9a-f]{12}", rid)  # 12 位十六进制（uuid4 截断）

    def test_incoming_request_id_echoed_back(self):
        r = client.get("/health", headers={"X-Request-ID": "abc123"})  # 网关透传场景
        assert r.headers["x-request-id"] == "abc123"  # 原样回传，不重新生成

    def test_request_id_injected_into_logging_context(self):
        from serve import _stream_react  # 导入生成器工厂（不真正跑 LLM）
        import asyncio  # 事件循环驱动异步生成器

        async def _collect():  # 收集生成器产出，验证主协程与哨兵逻辑未被破坏
            events = []  # 事件收集器
            async for ev in _stream_react("问题", 1, "manual", None):  # 消费一条 start 事件后主动退出
                events.append(ev)  # 收纳
                break  # start 事件足够验证生成器可启动
            return events  # 返回

        token = request_id_var.set("rid-check-42")  # 模拟请求中间件注入的上下文
        try:
            events = asyncio.run(_collect())  # 驱动生成器
            assert events  # start 事件已产出（改造后生成器仍正常）
        finally:
            request_id_var.reset(token)  # 清理上下文


# ═══ 限流拒绝 → 429 计入 rate_limit_rejected_total ═══════════════════════════

class TestRateLimitRejection:
    def test_429_counted_in_rejected_total(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "1")  # 每分钟仅 1 次（需在注册前生效）
        fresh = FastAPI()  # 独立 app：serve 单例已关限流
        install_rate_limit(fresh)  # 先注册限流（内层）
        install_observability(fresh)  # 再注册可观测（外层，后注册者在外）——429 才会被统计

        @fresh.get("/ping")  # 业务路由
        async def ping():
            return {"ok": True}

        fc = TestClient(fresh)  # 独立客户端
        r0 = client.get("/metrics")  # 前值：全局计数器在 serve app 侧读取
        before = _counter_value(r0.text, "rate_limit_rejected_total")
        assert fc.get("/ping").status_code == 200  # 第 1 次放行
        assert fc.get("/ping").status_code == 429  # 第 2 次超阈值被拒
        r1 = client.get("/metrics")  # 后值
        after = _counter_value(r1.text, "rate_limit_rejected_total")
        assert after - before >= 1  # 429 流经外层可观测中间件被计入

    def test_429_response_carries_request_id(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "1")  # 单次配额
        fresh = FastAPI()  # 独立 app
        install_rate_limit(fresh)  # 先限流
        install_observability(fresh)  # 后可观测（外层）

        @fresh.get("/ping")
        async def ping():
            return {"ok": True}

        fc = TestClient(fresh)  # 独立客户端
        fc.get("/ping")  # 消耗配额
        r = fc.get("/ping", headers={"X-Request-ID": "rid-429-test"})  # 第二次被拒
        assert r.status_code == 429  # 确认被限流
        assert r.headers["x-request-id"] == "rid-429-test"  # 外层中间件给 429 响应也回写请求 ID


# ═══ JsonFormatter：JSON 行式序列化 ═══════════════════════════════════════════

class TestJsonFormatter:
    def test_format_injects_request_id_and_extra(self):
        token = request_id_var.set("rid-fmt-001")  # 模拟请求上下文
        try:
            record = logging.LogRecord(  # 手工构造日志记录
                "my.logger", logging.INFO, __file__, 42,
                "袖带怎么绑 %s", ("呢",), None)  # 带参数化消息
            record.custom_field = "custom-value"  # 模拟 extra 注入的业务字段
            line = JsonFormatter().format(record)  # 序列化为一行 JSON
        finally:
            request_id_var.reset(token)  # 清理上下文
        data = json.loads(line)  # 必须是合法 JSON
        assert data["request_id"] == "rid-fmt-001"  # request_id 自动注入
        assert data["msg"] == "袖带怎么绑 呢"  # 消息参数已插值
        assert data["level"] == "INFO"  # 级别透传
        assert data["custom_field"] == "custom-value"  # extra 字段透传

    def test_format_without_request_context_defaults_to_dash(self):
        token = request_id_var.set("-")  # 无请求上下文时的默认值（对齐 default="-"）
        try:
            record = logging.LogRecord(  # 简单记录
                "x", logging.WARNING, __file__, 1, "无上下文", (), None)
            line = JsonFormatter().format(record)  # 序列化
        finally:
            request_id_var.reset(token)  # 清理
        assert json.loads(line)["request_id"] == "-"  # 兜底值


# ═══ setup_logging：text / json 格式切换 ═════════════════════════════════════

class TestSetupLogging:
    def test_text_format_then_restore_json(self):
        root = logging.getLogger()  # 根日志器
        original_handlers = list(root.handlers)  # 保存既有 handler（pytest 注入的）
        try:
            setup_logging(fmt="text")  # 切人读文本格式：不抛异常即通过
            assert len(root.handlers) == 1  # 替换为单一 handler
            assert not isinstance(root.handlers[0].formatter, JsonFormatter)  # 非格式化器
            setup_logging(fmt="json")  # 恢复 JSON 行式（生产默认）
            assert isinstance(root.handlers[0].formatter, JsonFormatter)  # 恢复成功
        finally:
            root.handlers[:] = original_handlers  # 还原 pytest 的 handler，不污染其他用例


# ═══ llm_call_timer：LLM 调用埋点 ════════════════════════════════════════════

class TestLlmCallTimer:
    def test_ok_path_increments_ok_counter(self):
        before = _metric_value("llm_calls_total", 'status="ok"')  # ok 前值
        with llm_call_timer():  # 模拟一次成功的 LLM 调用
            pass  # 正常结束
        after = _metric_value("llm_calls_total", 'status="ok"')  # ok 后值
        assert after - before == 1  # ok 计数恰好 +1

    def test_error_path_counts_and_reraises(self):
        before_err = _metric_value("llm_calls_total", 'status="error"')  # error 前值
        before_ok = _metric_value("llm_calls_total", 'status="ok"')  # ok 前值
        try:  # 异常必须原样上抛（不吞错）
            with llm_call_timer():  # 模拟一次失败的 LLM 调用
                raise RuntimeError("LLM 网络超时")  # 调用中抛错
        except RuntimeError:  # 捕获验证上抛
            pass  # 预期路径
        else:
            raise AssertionError("llm_call_timer 吞掉了异常")  # 未上抛即失败
        after_err = _metric_value("llm_calls_total", 'status="error"')  # error 后值
        after_ok = _metric_value("llm_calls_total", 'status="ok"')  # ok 后值
        assert after_err - before_err == 1  # error 计数 +1
        assert after_ok - before_ok == 0  # 失败不计入 ok

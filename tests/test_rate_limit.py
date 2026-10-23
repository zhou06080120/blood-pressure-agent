"""rate_limit 限流器测试：纯类逻辑（now 参数注入，零真实等待）+ FastAPI 中间件集成

设计要点：
  - 时间全部通过 now 参数注入构造，绝不真实 sleep —— 测试快速且稳定
  - 中间件测试自建独立 FastAPI 应用（serve 单例 app 在 conftest 已设
    RATE_LIMIT_RPM=0 关闭限流），monkeypatch 临时改环境变量再注册
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_limit import RateLimiter, install_rate_limit


# ═══ 纯类：滑动窗口逻辑（now 注入） ═══════════════════════════════════════════

class TestRateLimiter:
    def test_rpm3_allows_three_rejects_fourth(self):
        limiter = RateLimiter(rpm=3)  # 每窗口 3 次配额
        assert limiter.allow("ip", now=0) is True   # 第 1 次放行
        assert limiter.allow("ip", now=0) is True   # 第 2 次放行
        assert limiter.allow("ip", now=0) is True   # 第 3 次放行
        assert limiter.allow("ip", now=0) is False  # 第 4 次拒绝

    def test_window_slides(self):
        limiter = RateLimiter(rpm=2)  # 每窗口 2 次配额
        assert limiter.allow("ip", now=0) is True   # 窗口内第 1 次
        assert limiter.allow("ip", now=0) is True   # 窗口内第 2 次
        assert limiter.allow("ip", now=0) is False  # 窗口内已满
        assert limiter.allow("ip", now=61) is True  # 61s：0s 的记录已滑出窗口，重新放行

    def test_rpm0_disables_limiting(self):
        limiter = RateLimiter(rpm=0)  # 0 = 关闭限流
        for _ in range(10):  # 打多少次都放行
            assert limiter.allow("ip", now=0) is True

    def test_retry_after(self):
        limiter = RateLimiter(rpm=1)  # 每窗口 1 次配额
        limiter.allow("ip", now=100)  # 100s 放行一次（队列记录 [100]）
        assert limiter.retry_after("ip", now=100) == 60  # 最早记录 100+60 滑出窗口需等 60s

    def test_retry_after_unknown_key_returns_1(self):
        limiter = RateLimiter(rpm=1)
        assert limiter.retry_after("nobody", now=100) == 1  # 无记录：至少 1 秒

    def test_evict_stale_removes_expired_queues(self):
        limiter = RateLimiter(rpm=2)
        limiter._hits["stale"].append(1.0)    # 早已滑出窗口的旧记录
        limiter._hits["alive"].append(100.0)  # 窗口内的活跃记录
        limiter._evict_stale(now=100.0)       # cutoff = 100 - 60 = 40
        assert "stale" not in limiter._hits   # 过期队列被清理
        assert "alive" in limiter._hits       # 活跃队列保留


# ═══ 中间件集成：429 + Retry-After / 探针豁免 ═════════════════════════════════

def _make_app() -> FastAPI:
    """新建独立 FastAPI 应用并注册限流 + /ping /health 两路由（不污染 serve 单例 app）"""
    app = FastAPI()  # 独立实例：install_rate_limit 注册时读环境变量
    install_rate_limit(app)

    @app.get("/ping")  # 普通业务路由（受限流约束）
    async def ping():
        return {"ok": True}

    @app.get("/health")  # 探针路由（豁免路径）
    async def health():
        return {"status": "ok"}

    return app


class TestMiddleware:
    def test_429_after_threshold_with_retry_after_header(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "3")  # 需在 install_rate_limit 之前生效
        client = TestClient(_make_app())
        for _ in range(3):  # 前 3 次全部放行
            assert client.get("/ping").status_code == 200
        r = client.get("/ping")  # 第 4 次超阈值
        assert r.status_code == 429
        assert r.headers["retry-after"].isdigit()  # 告知等待秒数
        assert "频繁" in r.json()["detail"]  # 用户可读提示

    def test_health_exempt_from_limiting(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "3")
        client = TestClient(_make_app())
        for _ in range(6):  # 探针端点豁免：超过阈值仍全放行
            assert client.get("/health").status_code == 200

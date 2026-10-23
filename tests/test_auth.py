"""auth API Key 双级鉴权测试：401/403、豁免路径、角色写入、开发模式放行

设计要点：
  - 每个用例自建独立 FastAPI 应用 + monkeypatch.setenv，不污染 serve 单例 app
  - auth._auth_enabled 是模块级开关（require_admin 读取），用例前后必须复位 False，
    否则跨文件污染后续测试模块（serve 单例的 /cache/clear 会被 403 误拦）
"""

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

import auth
from auth import install_auth, require_admin


@pytest.fixture(autouse=True)
def _reset_auth_flag():
    """用例前后复位模块级鉴权开关（防 _auth_enabled=True 泄漏到其他测试模块）"""
    auth._auth_enabled = False  # 前置复位：上一用例异常退出也可能残留 True
    yield
    auth._auth_enabled = False  # 后置复位：常规清理


def _admin_guard(request: Request) -> None:
    """管理端点守卫依赖：与 serve.py 同款包装（require_admin 的 request 形参无类型注解，
    直接 Depends 会被 FastAPI 按查询参数解析成 422，需显式标注 Request 后转发）"""
    require_admin(request)


def _make_app() -> FastAPI:
    """新建独立应用：业务路由 + 管理路由（require_admin 守卫）+ 豁免路由 + 角色回显"""
    app = FastAPI()

    @app.get("/ping")  # 业务路由（任何有效 Key 可访问）
    async def ping():
        return {"ok": True}

    @app.get("/admin", dependencies=[Depends(_admin_guard)])  # 管理路由（仅 admin 角色）
    async def admin():
        return {"ok": True}

    @app.get("/whoami")  # 回显鉴权中间件写入的状态（角色/指纹）
    async def whoami(request: Request):
        return {"role": getattr(request.state, "api_role", None),
                "actor": getattr(request.state, "actor_id", None)}

    @app.get("/health")  # 豁免路径（探针）
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")  # 豁免路径（指标抓取）
    async def metrics():
        return {}

    @app.get("/")  # 豁免路径（前端页面）
    async def root():
        return {"page": True}

    return app


def _clear_key_env(monkeypatch):
    """清空两个 Key 环境变量（确保用例不受外部环境影响）"""
    monkeypatch.delenv("BP_API_KEYS", raising=False)
    monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)


# ═══ 开发模式：未配置 Key 不拦截 ═══════════════════════════════════════════════

class TestDisabledMode:
    def test_no_keys_returns_none_and_passes(self, monkeypatch):
        _clear_key_env(monkeypatch)
        assert install_auth(_make_app()) is None  # 未配置 Key：不注册中间件，返回 None
        r = TestClient(_make_app()).get("/ping")  # 不带任何 Key 头直访
        assert r.status_code == 200  # 本地开发模式放行

    def test_require_admin_passes_when_disabled(self, monkeypatch):
        _clear_key_env(monkeypatch)
        app = _make_app()
        install_auth(app)  # 未启用：install 返回 None，_auth_enabled 保持 False
        assert auth._auth_enabled is False  # 开关未打开
        assert TestClient(app).get("/admin").status_code == 200  # 管理端点同样放行


# ═══ 普通 Key：401 拦截与角色写入 ═════════════════════════════════════════════

class TestUserKeyAuth:
    def test_missing_or_wrong_key_401(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "user-key-1")
        monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        assert c.get("/ping").status_code == 401  # 缺 Key
        assert c.get("/ping", headers={"X-API-Key": "wrong"}).status_code == 401  # 错 Key（与缺 Key 同话术防探测）
        assert c.get("/ping", headers={"X-API-Key": "user-key-1"}).status_code == 200  # 对 Key 放行

    def test_multiple_keys_comma_separated(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "k1, k2 ,")  # 逗号分隔 + 空白项 + 尾逗号
        monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        assert c.get("/ping", headers={"X-API-Key": "k1"}).status_code == 200  # 第一个 Key
        assert c.get("/ping", headers={"X-API-Key": "k2"}).status_code == 200  # 第二个 Key（空白被解析掉）

    def test_role_and_actor_written_to_state(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "user-key-1")
        monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
        app = _make_app()
        install_auth(app)
        body = TestClient(app).get("/whoami", headers={"X-API-Key": "user-key-1"}).json()
        assert body["role"] == "user"  # 普通角色
        assert len(body["actor"]) == 8  # Key 指纹为 SHA-256 前 8 位（脱敏不落明文）


# ═══ 管理 Key：双级角色与 403 ═════════════════════════════════════════════════

class TestAdminKeyAuth:
    def test_admin_key_accesses_admin_endpoint(self, monkeypatch):
        monkeypatch.delenv("BP_API_KEYS", raising=False)
        monkeypatch.setenv("BP_ADMIN_KEYS", "admin-key-1")
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        assert c.get("/admin", headers={"X-API-Key": "admin-key-1"}).status_code == 200  # admin 放行
        assert c.get("/admin").status_code == 401  # 未认证直访管理端点：中间件先拦 401

    def test_user_key_forbidden_on_admin_endpoint(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "user-key-1")
        monkeypatch.setenv("BP_ADMIN_KEYS", "admin-key-1")
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        r = c.get("/admin", headers={"X-API-Key": "user-key-1"})  # 普通 Key 访问管理端点
        assert r.status_code == 403  # 角色不足（中间件放行后由 require_admin 拦截）
        assert "管理员" in r.json()["detail"]  # 用户可读提示

    def test_same_key_in_both_sets_admin_wins(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "shared-key")  # 同一 Key 同时出现在两个集合
        monkeypatch.setenv("BP_ADMIN_KEYS", "shared-key")
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        body = c.get("/whoami", headers={"X-API-Key": "shared-key"}).json()
        assert body["role"] == "admin"  # admin 覆盖 user（同 Key 不共存两级）
        assert c.get("/admin", headers={"X-API-Key": "shared-key"}).status_code == 200  # admin 权限生效


# ═══ 豁免路径与预检请求 ════════════════════════════════════════════════════════

class TestExemptPaths:
    def test_exempt_paths_pass_without_key(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "user-key-1")
        monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
        app = _make_app()
        install_auth(app)
        c = TestClient(app)
        for path in ("/health", "/metrics", "/"):  # 探针 + 前端页面豁免
            assert c.get(path).status_code == 200  # 不带 Key 也放行

    def test_options_preflight_passes(self, monkeypatch):
        monkeypatch.setenv("BP_API_KEYS", "user-key-1")
        monkeypatch.delenv("BP_ADMIN_KEYS", raising=False)
        app = _make_app()
        install_auth(app)
        r = TestClient(app).options("/ping")  # CORS 预检请求
        assert r.status_code != 401  # 预检放行（不被 Key 校验拦截）

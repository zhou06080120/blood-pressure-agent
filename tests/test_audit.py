"""audit 管理操作审计测试：JSONL 追加、字段完整、路径覆盖与目录自愈

全程通过 BP_AUDIT_PATH 定向到 tmp_path，不污染项目根 logs/ 目录。
"""

import json

from audit import audit_path, record_audit


def _redirect(tmp_path, monkeypatch, name="audit.jsonl"):
    """把审计日志重定向到临时目录，返回目标路径"""
    path = tmp_path / name
    monkeypatch.setenv("BP_AUDIT_PATH", str(path))
    return path


class TestRecordAudit:
    def test_appends_jsonl_lines(self, tmp_path, monkeypatch):
        path = _redirect(tmp_path, monkeypatch)
        record_audit("cache_clear", target="semantic_cache", detail={"n": 1}, actor="abcd1234")
        record_audit("cache_clear", target="semantic_cache", detail={"n": 0}, actor="abcd1234")
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # 追加式：两条记录各占一行
        assert json.loads(lines[0])["detail"] == {"n": 1}  # 首条内容完整
        assert json.loads(lines[1])["detail"] == {"n": 0}  # 次条内容完整

    def test_entry_fields_complete(self, tmp_path, monkeypatch):
        path = _redirect(tmp_path, monkeypatch)
        entry = record_audit("cache_clear", target="semantic_cache",
                             detail={"question_entries": 3}, actor="abcd1234")
        data = json.loads(path.read_text(encoding="utf-8"))  # 落盘内容
        assert entry == data  # 返回值与落盘一致
        for key in ("ts", "actor", "action", "target", "detail", "request_id"):  # 口径字段齐全
            assert key in data
        assert data["actor"] == "abcd1234"  # 操作者指纹原样落盘
        assert data["action"] == "cache_clear"  # 动作名
        assert data["detail"] == {"question_entries": 3}  # 附加信息

    def test_defaults_actor_dash_and_empty_target(self, tmp_path, monkeypatch):
        _redirect(tmp_path, monkeypatch)
        entry = record_audit("session_reset")  # 只给动作名，其余全缺省
        assert entry["actor"] == "-"  # 未鉴权默认 "-"
        assert entry["target"] == ""  # target 缺省空串
        assert entry["detail"] is None  # detail 缺省 None
        assert entry["request_id"] == "-"  # 无请求上下文时默认 "-"

    def test_chinese_content_preserved(self, tmp_path, monkeypatch):
        path = _redirect(tmp_path, monkeypatch)
        record_audit("灰度发布", target="检索链路", detail={"备注": "全量"}, actor="abcd1234")
        raw = path.read_text(encoding="utf-8")
        assert "灰度发布" in raw  # ensure_ascii=False：中文原样落盘不转义


class TestAuditPath:
    def test_env_override(self, tmp_path, monkeypatch):
        path = _redirect(tmp_path, monkeypatch, name="custom.jsonl")
        assert audit_path() == path  # BP_AUDIT_PATH 覆盖生效

    def test_default_path_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("BP_AUDIT_PATH", raising=False)
        assert audit_path().name == "audit.jsonl"  # 默认文件名
        assert audit_path().parent.name == "logs"  # 默认在项目根 logs/ 下

    def test_directory_autocreate(self, tmp_path, monkeypatch):
        path = tmp_path / "deep" / "nested" / "audit.jsonl"  # 父目录链不存在
        monkeypatch.setenv("BP_AUDIT_PATH", str(path))
        record_audit("x")  # 写入时自建目录
        assert path.exists()  # 目录自愈成功，文件已落盘

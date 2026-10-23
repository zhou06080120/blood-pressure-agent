"""react_function_calling FC 版 ReAct 测试：tool_calls 解析、并行执行、去重、消息回填"""

import react_function_calling as rfc
from react_function_calling import run

from conftest import FCClient, FCToolCall, fc_final, fc_tool_calls


def _calc_call(expr="1+1", call_id="call_1"):
    return FCToolCall(call_id, "calculator", f'{{"expr": "{expr}"}}')


class TestFinalPath:
    def test_direct_answer_with_disclaimer(self, monkeypatch):
        client = FCClient([fc_final("袖带下缘距肘窝2-3厘米。")])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("袖带怎么绑？"))
        final = events[-1]
        assert final["type"] == "final"
        assert "仅供参考" in final["answer"]
        assert final["guards"]["state"] == "final"

    def test_safety_gate_short_circuit(self, monkeypatch):
        client = FCClient([])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("帮我开点降压药"))
        assert events[0].get("safety_blocked") == "medication"
        assert client.calls == []

    def test_history_passed(self, monkeypatch):
        client = FCClient([fc_final("回答。")])
        monkeypatch.setattr(rfc, "client", client)
        history = [{"role": "user", "content": "旧问题"},
                   {"role": "assistant", "content": "旧回答"}]
        list(run("新问题", history=history))
        messages = client.calls[0]["messages"]
        assert messages[1]["content"] == "旧问题"
        assert messages[-1]["content"] == "新问题"


class TestToolCallsPath:
    def test_single_tool_then_final(self, monkeypatch):
        client = FCClient([
            fc_tool_calls(_calc_call("80 + (120-80)/3")),
            fc_final("平均动脉压约93.3。"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("平均动脉压？"))
        types = [e["type"] for e in events]
        assert types == ["action", "final"]
        assert events[0]["observation"] == "93.333333"
        assert events[0].get("parallel") == 1

    def test_parallel_tools_same_turn(self, monkeypatch):
        client = FCClient([
            fc_tool_calls(
                _calc_call("1+1", "call_1"),
                FCToolCall("call_2", "product_lookup", '{"model": "欧姆龙HEM-7156"}'),
            ),
            fc_final("两个工具的结果已汇总。"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("1+1以及HEM-7156规格"))
        actions = [e for e in events if e["type"] == "action"]
        assert len(actions) == 2
        assert {a["action"] for a in actions} == {"calculator", "product_lookup"}
        assert all(a.get("parallel") == 2 for a in actions)  # 同轮并行执行

    def test_unknown_tool(self, monkeypatch):
        client = FCClient([
            fc_tool_calls(FCToolCall("call_1", "no_such_tool", "{}")),
            fc_final("已了解。"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("任意问题"))
        assert "未知工具" in events[0]["observation"]

    def test_bad_json_arguments(self, monkeypatch):
        # 参数 JSON 非法 → 空 dict → 调用缺必填参数 → TypeError → "工具参数错误"
        client = FCClient([
            fc_tool_calls(FCToolCall("call_1", "calculator", "不是JSON")),
            fc_final("ok"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("任意问题"))
        assert "工具参数错误" in events[0]["observation"]

    def test_tool_message_written_back(self, monkeypatch):
        client = FCClient([
            fc_tool_calls(_calc_call("1+1", "call_9")),
            fc_final("done"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        list(run("x"))
        messages = client.calls[1]["messages"]
        # 对话中混有 SDK 消息对象（assistant）与 dict（tool），只取 dict 形态的 tool 消息
        tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_9"  # 与 tool_call.id 对齐

    def test_duplicate_blocked_across_turns(self, monkeypatch):
        client = FCClient([
            fc_tool_calls(_calc_call("1+1", "call_1")),
            fc_tool_calls(_calc_call("1+1", "call_2")),   # 同工具同参数第二轮
            fc_tool_calls(_calc_call("1+1", "call_3")),   # 第三轮 → 阻断
            fc_final("2"),
        ])
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("一加一？", max_steps=5))
        actions = [e for e in events if e["type"] == "action"]
        assert len(actions) == 3
        assert actions[2].get("duplicate_blocked") is True
        assert "重复" in actions[2]["observation"]

    def test_max_steps_fallback(self, monkeypatch):
        script = [fc_tool_calls(_calc_call(f"{i}+{i}", f"call_{i}")) for i in range(2)]
        client = FCClient(script)
        monkeypatch.setattr(rfc, "client", client)
        events = list(run("x", max_steps=2))
        assert events[-1]["type"] == "max_steps"

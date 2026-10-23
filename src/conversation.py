"""
多轮对话会话管理 —— 血压计客服 Agent

教学重点：
  1. 会话隔离：每个 session_id 独立维护对话历史，互不干扰
  2. 历史拼接：ReAct 单轮的 Thought/Action/Observation 是临时的，
     跨轮只保留 user 问题 + assistant 最终回答，避免上下文膨胀
  3. 上下文窗口管控：超过 max_history 条消息时自动截断保留最近的对话
  4. 会话过期清理：避免内存无限增长
  5. 存储可插拔（P4）：会话记录委托 ConversationStore 读写——默认进程内存，
     配置 BP_REDIS_URL 后外置 Redis（多副本共享会话，水平扩展解锁）

使用方式：
  from conversation import conv_manager
  session_id = conv_manager.create_session()
  history = conv_manager.get_history(session_id)
  conv_manager.add_message(session_id, "user", "袖带怎么绑？")
  conv_manager.add_message(session_id, "assistant", "袖带应绑在...")
"""

# ═══ 导入区：time 记活跃时间、secrets 生成会话 id、Store 抽象与工厂 ═══
import time  # 时间戳：会话最后活跃时间与过期判定
import secrets  # 密码学安全随机数：生成不可预测的 session_id
from typing import Optional  # 元信息缺失时返回 Optional[dict]

from conversation_store import ConversationStore, create_store  # 会话存储抽象 + 按环境变量选择的工厂


# ═══ 功能块：会话管理器 —— 业务逻辑层，含隔离/截断/过期清理三重机制 ═══
class ConversationManager:
    """
    多轮对话会话管理器（存储可插拔）

    业务规则（截断上限/过期 TTL/消息投影）留在本层；
    记录读写委托 ConversationStore（内存或 Redis，线程安全由 Store 自身保证）。

    每个 session 维护一条 messages 列表，元素格式：
      {"role": "user"|"assistant", "content": str, "timestamp": float}
    """

    def __init__(self, max_history: int = 20, session_ttl: int = 1800,
                 store: ConversationStore | None = None):
        """
        :param max_history: 单个会话最多保留的消息条数，超出后截断保留最近部分
        :param session_ttl: 会话过期时间（秒），默认 30 分钟无活动自动清理
        :param store: 会话存储实现（缺省经 create_store() 决定：BP_REDIS_URL 配置则 Redis，否则内存）
        """
        self._store = store if store is not None else create_store()  # 存储实现：注入优先，否则按环境变量选择
        self._max_history = max_history  # 单会话消息条数上限
        self._session_ttl = session_ttl  # 会话闲置过期阈值（秒）

    # ── 会话生命周期 ──────────────────────────────────────────────────────────

    # ═══ 创建新会话：生成随机 id 并登记空消息列表 ═══
    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = secrets.token_hex(8)  # 16位十六进制，足够避免碰撞
        self._store.put(session_id, {"messages": [],  # 新会话消息列表为空
                                     "last_active": time.time()})  # 记录创建时刻为最后活跃时间
        return session_id  # 把 id 返回给前端保存

    # ═══ 清空历史：保留会话壳，只重置消息与活跃时间 ═══
    def clear(self, session_id: str) -> bool:
        """清空指定会话的历史，会话本身保留"""
        return self._store.clear(session_id)  # 委托存储清空（不存在返回 False）

    # ═══ 彻底销毁会话：从存储中移除（与 clear 的区别） ═══
    def destroy(self, session_id: str) -> bool:
        """彻底销毁会话"""
        return self._store.delete(session_id)  # 委托存储删除（不存在返回 False）

    # ═══ 过期清理：回收超时未活跃的会话，防止存储无限增长 ═══
    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.time()  # 当前时间
        expired = []  # 过期会话 id 收集器
        for sid in self._store.list_sessions():  # 遍历全部会话（存储层已做并发保护）
            rec = self._store.get(sid)  # 取会话记录（清理过程中被并发删除则跳过）
            if rec and now - rec["last_active"] > self._session_ttl:  # 闲置超过 TTL 即过期
                expired.append(sid)  # 登记待清理
        for sid in expired:  # 逐个移除
            self._store.delete(sid)  # 删除（幂等：已不存在也无害）
        return len(expired)  # 返回本次清理数量

    # ── 历史读写 ──────────────────────────────────────────────────────────────

    # ═══ 追加消息：写入会话历史并按 max_history 截断 ═══
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        追加一条消息到会话历史
        :param role: "user" 或 "assistant"
        :param content: 消息文本（assistant 存最终回答，不含中间Thought/Action步骤）
        """
        return self._store.append(session_id, role, content,  # 委托存储追加（自动盖时间戳+刷新活跃）
                                  max_history=self._max_history)  # 截断上限由本层业务规则传入

    # ═══ 读取历史：转成 OpenAI messages 格式给 ReAct 循环拼上下文 ═══
    def get_history(self, session_id: str) -> list[dict]:
        """
        获取会话历史，返回 OpenAI messages 格式：
          [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        rec = self._store.get(session_id)  # 取会话记录
        if rec is None:  # 会话不存在
            return []  # 空历史（按单轮处理）
        return [  # 投影成精简格式（去掉 timestamp）
            {"role": m["role"], "content": m["content"]}
            for m in rec["messages"]
        ]

    # ═══ 会话元信息：消息数、活跃时间与最近消息摘要 ═══
    def get_info(self, session_id: str) -> Optional[dict]:
        """获取会话元信息：消息数、最后活跃时间"""
        rec = self._store.get(session_id)  # 取会话记录
        if rec is None:  # 会话不存在
            return None  # 明确返回 None 供路由层报错
        return {
            "session_id": session_id,  # 会话 id
            "message_count": len(rec["messages"]),  # 消息条数
            "last_active": rec["last_active"],  # 最后活跃时间戳
            "messages": [  # 消息预览列表
                {"role": m["role"], "content": m["content"][:100], "timestamp": m["timestamp"]}  # 正文截断 100 字
                for m in rec["messages"]
            ],
        }

    # ═══ 列出全部会话摘要：供管理/调试接口使用 ═══
    def list_sessions(self) -> list[dict]:
        """列出所有会话摘要（用于管理/调试）"""
        now = time.time()  # 当前时间
        out = []  # 摘要收集器
        for sid in self._store.list_sessions():  # 遍历全部会话 id
            rec = self._store.get(sid)  # 取会话记录（遍历中被并发删除则跳过）
            if rec is None:  # 已不存在
                continue  # 跳过
            out.append({
                "session_id": sid,  # 会话 id
                "message_count": len(rec["messages"]),  # 消息数
                "last_active": rec["last_active"],  # 最后活跃时间
                "idle_seconds": round(now - rec["last_active"], 0),  # 闲置秒数
            })
        return out  # 返回摘要列表


# 全局单例，serve.py 和 agent.py 共用
conv_manager = ConversationManager(max_history=20, session_ttl=1800)  # 模块级唯一实例：20 条消息上限 / 30 分钟过期；存储按 BP_REDIS_URL 自动选择（未配置=内存，零行为变化）

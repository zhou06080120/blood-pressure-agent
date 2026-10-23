"""
多轮对话会话管理 —— 血压计客服 Agent

教学重点：
  1. 会话隔离：每个 session_id 独立维护对话历史，互不干扰
  2. 历史拼接：ReAct 单轮的 Thought/Action/Observation 是临时的，
     跨轮只保留 user 问题 + assistant 最终回答，避免上下文膨胀
  3. 上下文窗口管控：超过 max_history 条消息时自动截断保留最近的对话
  4. 会话过期清理：避免内存无限增长

使用方式：
  from conversation import conv_manager
  session_id = conv_manager.create_session()
  history = conv_manager.get_history(session_id)
  conv_manager.add_message(session_id, "user", "袖带怎么绑？")
  conv_manager.add_message(session_id, "assistant", "袖带应绑在...")
"""

# ═══ 导入区：time 记活跃时间、threading 加锁、secrets 生成会话 id ═══
import time  # 时间戳：会话最后活跃时间与过期判定
import threading  # Lock：多请求并发读写会话时的线程安全保护
import secrets  # 密码学安全随机数：生成不可预测的 session_id
from typing import Optional  # 元信息缺失时返回 Optional[dict]


# ═══ 功能块：会话管理器 —— 内存级存储，含隔离/截断/过期清理三重机制 ═══
class ConversationManager:
    """
    内存级多轮对话会话管理器

    每个 session 维护一条 messages 列表，元素格式：
      {"role": "user"|"assistant", "content": str, "timestamp": float}
    """

    def __init__(self, max_history: int = 20, session_ttl: int = 1800):
        """
        :param max_history: 单个会话最多保留的消息条数，超出后截断保留最近部分
        :param session_ttl: 会话过期时间（秒），默认 30 分钟无活动自动清理
        """
        self._sessions: dict[str, dict] = {}   # session_id -> {"messages": [...], "last_active": float}
        self._max_history = max_history  # 单会话消息条数上限
        self._session_ttl = session_ttl  # 会话闲置过期阈值（秒）
        self._lock = threading.Lock()  # 全局锁：FastAPI 多 worker 线程并发下的安全保护

    # ── 会话生命周期 ──────────────────────────────────────────────────────────

    # ═══ 创建新会话：生成随机 id 并登记空消息列表 ═══
    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = secrets.token_hex(8)  # 16位十六进制，足够避免碰撞
        with self._lock:  # 加锁写入共享字典
            self._sessions[session_id] = {
                "messages": [],  # 新会话消息列表为空
                "last_active": time.time(),  # 记录创建时刻为最后活跃时间
            }
        return session_id  # 把 id 返回给前端保存

    # ═══ 内部辅助：刷新会话最后活跃时间（过期判定依据） ═══
    def _touch(self, session_id: str):
        """更新会话最后活跃时间"""
        if session_id in self._sessions:  # 会话存在才更新
            self._sessions[session_id]["last_active"] = time.time()  # 刷成当前时间

    # ═══ 清空历史：保留会话壳，只重置消息与活跃时间 ═══
    def clear(self, session_id: str) -> bool:
        """清空指定会话的历史，会话本身保留"""
        with self._lock:  # 加锁修改
            if session_id in self._sessions:  # 会话存在
                self._sessions[session_id]["messages"] = []  # 清空消息列表
                self._sessions[session_id]["last_active"] = time.time()  # 刷新活跃时间
                return True  # 成功
            return False  # 会话不存在，失败

    # ═══ 彻底销毁会话：从字典中移除（与 clear 的区别） ═══
    def destroy(self, session_id: str) -> bool:
        """彻底销毁会话"""
        with self._lock:  # 加锁删除
            if session_id in self._sessions:  # 存在才删
                del self._sessions[session_id]  # 移除整个会话记录
                return True
            return False  # 不存在则失败

    # ═══ 过期清理：回收超时未活跃的会话，防止内存无限增长 ═══
    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.time()  # 当前时间
        expired = [  # 先在锁外筛选过期 id（减少锁内工作）
            sid for sid, info in self._sessions.items()  # 遍历全部会话
            if now - info["last_active"] > self._session_ttl  # 闲置超过 TTL 即过期
        ]
        with self._lock:  # 加锁执行删除
            for sid in expired:  # 逐个移除
                self._sessions.pop(sid, None)  # pop 带默认值：不存在也不报错
        return len(expired)  # 返回本次清理数量

    # ── 历史读写 ──────────────────────────────────────────────────────────────

    # ═══ 追加消息：写入会话历史并按 max_history 截断 ═══
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        追加一条消息到会话历史
        :param role: "user" 或 "assistant"
        :param content: 消息文本（assistant 存最终回答，不含中间Thought/Action步骤）
        """
        with self._lock:  # 加锁写入
            if session_id not in self._sessions:  # 会话不存在
                return False  # 写入失败
            self._sessions[session_id]["messages"].append({
                "role": role,  # 角色：user/assistant
                "content": content,  # 消息正文
                "timestamp": time.time(),  # 写入时间戳
            })
            # 超出上限时截断，保留最近 max_history 条
            msgs = self._sessions[session_id]["messages"]  # 取出消息列表
            if len(msgs) > self._max_history:  # 超上限
                self._sessions[session_id]["messages"] = msgs[-self._max_history:]  # 只保留最近 N 条
            self._sessions[session_id]["last_active"] = time.time()  # 追加消息视作活跃，刷新时间
            return True  # 写入成功

    # ═══ 读取历史：转成 OpenAI messages 格式给 ReAct 循环拼上下文 ═══
    def get_history(self, session_id: str) -> list[dict]:
        """
        获取会话历史，返回 OpenAI messages 格式：
          [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        with self._lock:  # 加锁读取
            if session_id not in self._sessions:  # 会话不存在
                return []  # 空历史（按单轮处理）
            return [  # 投影成精简格式（去掉 timestamp）
                {"role": m["role"], "content": m["content"]}
                for m in self._sessions[session_id]["messages"]
            ]

    # ═══ 会话元信息：消息数、活跃时间与最近消息摘要 ═══
    def get_info(self, session_id: str) -> Optional[dict]:
        """获取会话元信息：消息数、最后活跃时间"""
        with self._lock:  # 加锁读取
            if session_id not in self._sessions:  # 会话不存在
                return None  # 明确返回 None 供路由层报错
            info = self._sessions[session_id]  # 取会话数据
            return {
                "session_id": session_id,  # 会话 id
                "message_count": len(info["messages"]),  # 消息条数
                "last_active": info["last_active"],  # 最后活跃时间戳
                "messages": [  # 消息预览列表
                    {"role": m["role"], "content": m["content"][:100], "timestamp": m["timestamp"]}  # 正文截断 100 字
                    for m in info["messages"]
                ],
            }

    # ═══ 列出全部会话摘要：供管理/调试接口使用 ═══
    def list_sessions(self) -> list[dict]:
        """列出所有会话摘要（用于管理/调试）"""
        with self._lock:  # 加锁遍历
            now = time.time()  # 当前时间
            return [
                {
                    "session_id": sid,  # 会话 id
                    "message_count": len(info["messages"]),  # 消息数
                    "last_active": info["last_active"],  # 最后活跃时间
                    "idle_seconds": round(now - info["last_active"], 0),  # 闲置秒数
                }
                for sid, info in self._sessions.items()
            ]


# 全局单例，serve.py 和 agent.py 共用
conv_manager = ConversationManager(max_history=20, session_ttl=1800)  # 模块级唯一实例：20 条消息上限 / 30 分钟过期

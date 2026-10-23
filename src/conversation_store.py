"""
会话存储抽象与实现 —— P4 水平扩展基建

把 ConversationManager 原先进程内存 dict 的会话存储抽成可插拔 Store：
  - ConversationStore        抽象基类：get / append / clear / delete / put / list_sessions
  - MemoryConversationStore  进程内存实现（默认）：原行为原样搬迁，零外部依赖
  - RedisConversationStore   redis-py 实现：读 BP_REDIS_URL，多实例共享会话（水平扩展解锁）

选择策略（create_store 工厂）：
  - BP_REDIS_URL 未配置            → Memory（零行为变化，本地开发/单实例默认）
  - BP_REDIS_URL 已配置且可连接     → Redis（多副本部署共享会话状态）
  - BP_REDIS_URL 配了但连不上/缺包  → 自动回落 Memory 并 logger.warning（启动时一次，服务不阻塞）

记录形态（与原 ConversationManager._sessions 的 value 一致）：
  {"messages": [{"role": "user|assistant", "content": str, "timestamp": float}, ...],
   "last_active": float}
"""

import json       # Redis 实现的记录序列化
import logging    # 回落告警与外置提示日志
import os         # 读 BP_REDIS_URL 配置
import threading  # Memory 实现的并发读写锁
import time       # 消息时间戳与活跃时间
from abc import ABC, abstractmethod  # 抽象基类定义
from typing import Optional  # 元信息缺失时返回 Optional[dict]

logger = logging.getLogger(__name__)  # 本模块日志器


# ═══ 功能块：抽象基类 —— 会话记录的读写契约 ═══
class ConversationStore(ABC):
    """会话存储抽象：上层 ConversationManager 只依赖本接口，存储实现可插拔。

    记录形态：{"messages": [{"role","content","timestamp"}...], "last_active": float}
    """

    @abstractmethod
    def get(self, session_id: str) -> Optional[dict]:
        """取整条会话记录；会话不存在返回 None。"""

    @abstractmethod
    def put(self, session_id: str, record: dict) -> None:
        """整体写入会话记录（创建新会话 / 更新既有会话）。"""

    @abstractmethod
    def append(self, session_id: str, role: str, content: str,
               max_history: int | None = None) -> bool:
        """追加一条消息（自动盖时间戳）并刷新 last_active；会话不存在返回 False。

        max_history 提供时按存储自身机制截断保留最近 N 条（防上下文无限膨胀）。
        """

    @abstractmethod
    def clear(self, session_id: str) -> bool:
        """清空消息但保留会话壳（刷新 last_active）；会话不存在返回 False。"""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """彻底删除会话记录；会话不存在返回 False。"""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """列出全部会话 id（供过期清理与调试接口遍历）。"""


# ═══ 功能块：内存实现 —— 原 ConversationManager 行为原样搬迁 ═══
class MemoryConversationStore(ConversationStore):
    """进程内存实现（默认）：dict + 锁，行为与原 _sessions 完全一致。"""

    def __init__(self):
        self._sessions: dict[str, dict] = {}  # session_id -> {"messages": [...], "last_active": float}
        self._lock = threading.Lock()  # FastAPI 多 worker 线程并发下的安全保护

    def get(self, session_id: str) -> Optional[dict]:
        """取会话记录（返回浅拷贝：messages 列表复制，防调用方原地篡改内部状态）"""
        with self._lock:  # 加锁读取
            rec = self._sessions.get(session_id)  # 不存在得 None
            if rec is None:  # 会话不存在
                return None  # 明确返回 None
            return {"messages": list(rec["messages"]),  # 消息列表浅拷贝（消息 dict 本身只读使用）
                    "last_active": rec["last_active"]}  # 活跃时间原样带出

    def put(self, session_id: str, record: dict) -> None:
        """整体写入会话记录（入参做拷贝，隔离调用方后续修改）"""
        with self._lock:  # 加锁写入
            self._sessions[session_id] = {"messages": list(record.get("messages", [])),  # 消息列表拷贝入存
                                          "last_active": record.get("last_active", time.time())}  # 活跃时间缺省当前

    def append(self, session_id: str, role: str, content: str,
               max_history: int | None = None) -> bool:
        """追加消息并截断（超上限只保留最近 max_history 条），刷新活跃时间"""
        with self._lock:  # 加锁修改
            rec = self._sessions.get(session_id)  # 取目标会话
            if rec is None:  # 会话不存在
                return False  # 写入失败
            rec["messages"].append({"role": role, "content": content,  # 消息三元组
                                    "timestamp": time.time()})  # 写入时间戳
            if max_history and len(rec["messages"]) > max_history:  # 超出上限
                rec["messages"] = rec["messages"][-max_history:]  # 截断保留最近 N 条
            rec["last_active"] = time.time()  # 追加消息视作活跃
            return True  # 写入成功

    def clear(self, session_id: str) -> bool:
        """清空消息保留会话壳，刷新活跃时间"""
        with self._lock:  # 加锁修改
            rec = self._sessions.get(session_id)  # 取目标会话
            if rec is None:  # 会话不存在
                return False  # 清空失败
            rec["messages"] = []  # 消息列表置空（会话壳保留）
            rec["last_active"] = time.time()  # 刷新活跃时间
            return True  # 清空成功

    def delete(self, session_id: str) -> bool:
        """彻底删除会话记录"""
        with self._lock:  # 加锁删除
            return self._sessions.pop(session_id, None) is not None  # pop 带默认值：不存在返回 False

    def list_sessions(self) -> list[str]:
        """列出全部会话 id（快照拷贝，防遍历期间并发修改）"""
        with self._lock:  # 加锁遍历
            return list(self._sessions.keys())  # 键列表拷贝


# ═══ 功能块：Redis 实现 —— 会话状态外置，多副本共享（水平扩展） ═══
class RedisConversationStore(ConversationStore):
    """redis-py 实现：每个会话一个 String 键存 JSON 记录，命名空间隔离。

    redis 包惰性导入（未安装时 import 才报错，由 create_store 捕获回落 Memory）；
    构造时 ping 探活，连不上立即抛出（同样由 create_store 回落）。
    测试可通过 client= 参数注入 fakeredis，无需真实 Redis。
    """

    def __init__(self, url: str | None = None, client=None, namespace: str = "bp:conv"):
        """
        :param url:       Redis 连接串（缺省读 BP_REDIS_URL）
        :param client:    已构造的客户端（测试注入 fakeredis 用；优先于 url）
        :param namespace: 键名前缀（多服务共用同一 Redis 实例时的隔离空间）
        """
        self._namespace = namespace  # 键前缀
        if client is not None:  # 外部注入客户端（测试路径）
            self._r = client  # 直接使用
        else:  # 自行构造连接
            import redis  # 惰性导入：未安装 redis 包时此处才抛 ImportError
            self._r = redis.Redis.from_url(url or os.getenv("BP_REDIS_URL", ""),  # 连接串
                                           decode_responses=True)  # 返回 str 而非 bytes，省逐次解码
        self._r.ping()  # 连接探活：失败立即上抛，交由 create_store 决定回落

    def _key(self, session_id: str) -> str:
        """会话 id → 带命名空间的 Redis 键名"""
        return f"{self._namespace}:{session_id}"  # 前缀:会话id

    def get(self, session_id: str) -> Optional[dict]:
        """取会话记录（JSON 反序列化）；不存在返回 None"""
        raw = self._r.get(self._key(session_id))  # 读原始 JSON 串
        return json.loads(raw) if raw else None  # 反序列化（键不存在得 None）

    def put(self, session_id: str, record: dict) -> None:
        """整体写入会话记录（JSON 序列化）"""
        self._r.set(self._key(session_id),  # 目标键
                    json.dumps(record, ensure_ascii=False))  # 中文原样保留

    def append(self, session_id: str, role: str, content: str,
               max_history: int | None = None) -> bool:
        """读改写追加消息（单实例低并发场景足够；截断在写回前完成）"""
        rec = self.get(session_id)  # 先读现记录
        if rec is None:  # 会话不存在
            return False  # 写入失败
        rec.setdefault("messages", []).append({"role": role, "content": content,  # 追加消息三元组
                                               "timestamp": time.time()})  # 写入时间戳
        if max_history and len(rec["messages"]) > max_history:  # 超出上限
            rec["messages"] = rec["messages"][-max_history:]  # 截断保留最近 N 条
        rec["last_active"] = time.time()  # 刷新活跃时间
        self.put(session_id, rec)  # 整体写回
        return True  # 写入成功

    def clear(self, session_id: str) -> bool:
        """清空消息保留会话壳，刷新活跃时间"""
        rec = self.get(session_id)  # 先读现记录
        if rec is None:  # 会话不存在
            return False  # 清空失败
        rec["messages"] = []  # 消息置空（会话壳保留）
        rec["last_active"] = time.time()  # 刷新活跃时间
        self.put(session_id, rec)  # 整体写回
        return True  # 清空成功

    def delete(self, session_id: str) -> bool:
        """彻底删除会话记录"""
        return bool(self._r.delete(self._key(session_id)))  # DEL 返回删除键数（0=原本不存在）

    def list_sessions(self) -> list[str]:
        """SCAN 遍历命名空间下的全部会话 id（不阻塞 Redis，生产安全）"""
        prefix = f"{self._namespace}:"  # 匹配前缀
        ids = []  # 会话 id 收集器
        for key in self._r.scan_iter(match=f"{prefix}*"):  # SCAN 增量遍历（不阻塞服务）
            if isinstance(key, bytes):  # 客户端未开 decode_responses 时键为 bytes（如注入的 fakeredis）
                key = key.decode()  # 统一转 str
            ids.append(key[len(prefix):])  # 剥前缀得会话 id
        return ids  # 返回会话 id 列表


# ═══ 功能块：存储工厂 —— 按 BP_REDIS_URL 选择实现，失败回落内存 ═══
def create_store() -> ConversationStore:
    """按环境变量构造会话存储：

    - BP_REDIS_URL 未配置 → MemoryConversationStore（默认，零行为变化）
    - 已配置且可连接     → RedisConversationStore（会话外置，水平扩展就绪）
    - 已配置但不可用     → 回落 Memory 并 warning 一次（服务照常启动，单实例模式）
    """
    url = os.getenv("BP_REDIS_URL", "").strip()  # 连接串（去空白）
    if not url:  # 未配置：默认内存存储
        return MemoryConversationStore()  # 与原行为完全一致
    try:  # 配置了 Redis：尝试构造并探活
        store = RedisConversationStore(url=url)  # 内部惰性导入 redis 包 + ping 探活
        logger.info("会话存储已外置到 Redis（多副本水平扩展就绪）")  # 外置成功提示
        return store  # Redis 实现
    except Exception as e:  # 未安装 redis 包 / 连接失败 / 认证失败等
        logger.warning("Redis 会话存储不可用（%s），回落进程内存存储（单实例模式）", e)  # 启动时告警一次
        return MemoryConversationStore()  # 自动回落，服务不阻塞

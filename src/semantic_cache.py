"""
语义缓存（GPTCache 风格）—— 血压计客服

两级缓存，针对高频客服问题降低 LLM 往返延迟（目标 30%+ 降幅）：
  1. 问题级缓存 SemanticCache：
     - 精确命中：归一化后问题文本完全相同 → 直接返回缓存答案，0 次 LLM 调用
     - 语义命中：新问题与缓存问题的嵌入余弦相似度 ≥ 阈值 → 复用答案（向量缓存）
  2. 工具级缓存 ToolCache：
     - faq_search / troubleshoot / product_lookup 等确定性工具的结果缓存，
       多轮对话中重复检索同一内容时跳过检索重建

线程安全 + TTL 过期 + 命中率统计（/cache/stats 可观测）。

环境变量：
  BP_CACHE_TTL    缓存有效期（秒），默认 600
  BP_CACHE_SIM    语义命中相似度阈值，默认 0.92
"""

# ═══ 导入区：文本归一化/嵌入/加锁/哈希等基础设施 ═══
import os  # 读缓存配置环境变量
import re  # 归一化问题文本时剔除标点空白
import json  # 工具缓存 key 的参数序列化
import time  # TTL 过期判断的时间戳
import threading  # Lock 保证并发安全
import hashlib  # 工具缓存键的 MD5 哈希
import logging  # 嵌入失败等降级日志
from typing import Any  # 缓存值类型标注

logger = logging.getLogger(__name__)  # 本模块日志器

CACHE_TTL     = int(os.getenv("BP_CACHE_TTL", "600"))  # 条目有效期（秒），默认 10 分钟
CACHE_SIM     = float(os.getenv("BP_CACHE_SIM", "0.92"))  # 语义命中余弦阈值（越高越严格）
CACHE_MAX     = int(os.getenv("BP_CACHE_MAX", "2000"))  # 最大条目数，超出淘汰一半


# ═══ 功能块：问题文本归一化 —— 精确命中前的预处理 ═══
def _normalize(text: str) -> str:
    """归一化问题文本：去空白/标点/符号，用于精确命中"""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()  # 删掉所有非字母数字字符并转小写："袖带 怎么绑？"→"袖带怎么绑"


# ═══ 功能块：问题向量化 —— 复用 tools 的 embedding，失败降级为仅精确命中 ═══
def _embed(text: str):
    """问题嵌入（复用 tools 的 embedding），失败返回 None 降级为仅精确命中"""
    try:  # 嵌入模型可能未装/未配置，全部异常都吞掉降级
        from tools import _embed_query  # 复用统一的查询嵌入入口
        vec = _embed_query(text)  # 编码
        if vec is not None:  # 编码成功
            import numpy as np  # 局部导入 numpy（只有拿到向量才需要）
            return np.asarray(vec, dtype="float32").reshape(-1)  # 摊平成一维向量便于相似度计算
    except Exception as e:  # 任何异常
        logger.debug(f"cache 嵌入失败: {e}")  # debug 级别：不影响主流程
    return None  # 降级信号：调用方只走精确命中


# ═══ 功能块：余弦相似度 —— 语义命中的度量 ═══
def _cosine(a, b) -> float:
    """计算两个向量的余弦相似度，分母为 0 时返回 0（避免除零异常）"""
    import numpy as np  # 局部导入避免模块加载强依赖 numpy
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))  # 模长乘积做分母
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0  # 零向量防护：返回 0


# ═══ 功能块：问题级缓存 —— 精确命中 + 语义命中两级查找 ═══
class SemanticCache:
    """GPTCache 风格的问题级语义缓存"""

    def __init__(self, ttl: int | None = None, sim_threshold: float | None = None):
        """初始化问题级缓存。

        :param ttl: 条目有效期（秒），None 时用环境变量 BP_CACHE_TTL
        :param sim_threshold: 语义命中相似度阈值，None 时用环境变量 BP_CACHE_SIM
        """
        self.ttl = ttl or CACHE_TTL  # 本实例 TTL
        self.sim_threshold = sim_threshold or CACHE_SIM  # 本实例语义阈值
        self._lock = threading.Lock()  # 读写锁
        self._entries: dict[str, dict] = {}   # norm_key -> {"ts", "question", "answer", "vec"}
        self._hits = 0  # 总命中数（精确+语义）
        self._misses = 0  # 未命中数
        self._semantic_hits = 0  # 语义命中数（单独统计）

    def get(self, question: str) -> tuple[Any, str] | None:
        """命中返回 (cached_value, hit_type)，hit_type ∈ exact|semantic；未命中返回 None"""
        norm = _normalize(question)  # 先归一化作为查询键
        now = time.time()  # 当前时间戳（TTL 判定）
        with self._lock:  # 加锁保护共享字典
            # 1) 精确命中
            ent = self._entries.get(norm)  # 查归一化键
            if ent and now - ent["ts"] < self.ttl:  # 存在且未过期
                self._hits += 1  # 命中计数
                return ent["answer"], "exact"  # 直接返回缓存答案 + 精确命中标记
            # 2) 语义命中（向量相似）
            vec = _embed(question)  # 把新问题编码成向量（锁内执行有开销但保证一致）
            if vec is not None:  # 拿到向量才能比对
                best, best_sim = None, 0.0  # 最佳候选与相似度
                for norm_k, e in self._entries.items():  # 遍历所有缓存条目
                    if now - e["ts"] >= self.ttl:  # 跳过过期条目
                        continue
                    if e.get("vec") is not None and len(e["vec"]) == len(vec):  # 维度一致才可比
                        sim = _cosine(vec, e["vec"])  # 算余弦相似度
                        if sim > best_sim:  # 更新最佳候选
                            best, best_sim = e, sim
                if best is not None and best_sim >= self.sim_threshold:  # 最佳相似度达阈值 → 语义命中
                    self._hits += 1  # 计入总命中
                    self._semantic_hits += 1  # 计入语义命中
                    return best["answer"], "semantic"  # 复用相似问题的答案
            self._misses += 1  # 两级都未命中
            return None  # 明确返回 None，调用方走完整 ReAct

    def put(self, question: str, answer: Any):
        """写入一条问答缓存（含嵌入向量，供后续语义命中比对），超容量时淘汰最旧一半"""
        norm = _normalize(question)  # 归一化键
        with self._lock:  # 加锁写入
            self._entries[norm] = {"ts": time.time(), "question": question,
                                   "answer": answer, "vec": _embed(question)}  # 存时间戳/原文/答案/问题向量
            if len(self._entries) > CACHE_MAX:  # 超容量
                # 淘汰最旧的一半（简单 LRU-by-time）
                items = sorted(self._entries.items(), key=lambda kv: kv[1]["ts"])  # 按时间戳升序
                for k, _ in items[:CACHE_MAX // 2]:  # 最旧的一半
                    self._entries.pop(k, None)  # 移除

    def stats(self) -> dict:
        """返回问题级缓存命中率统计（命中/未命中/语义命中/条目数）"""
        total = self._hits + self._misses or 1  # 防除零：0 次查询时按 1 计
        return {"hits": self._hits, "misses": self._misses,
                "semantic_hits": self._semantic_hits,
                "hit_rate": round(self._hits / total, 3),
                "entries": len(self._entries)}  # 返回完整统计字典


# ═══ 功能块：工具级缓存 —— 确定性工具结果按 (工具,参数) 哈希缓存 ═══
class ToolCache:
    """工具结果缓存：确定性工具（检索/查询类）结果按参数缓存"""

    def __init__(self, ttl: int | None = None):
        """初始化工具结果缓存，ttl 为条目的有效期（秒）"""
        self.ttl = ttl or CACHE_TTL  # 本实例 TTL
        self._lock = threading.Lock()  # 读写锁
        self._entries: dict[str, dict] = {}  # md5键 -> {"ts","result","tool"}
        self._hits = 0  # 命中数
        self._misses = 0  # 未命中数

    @staticmethod
    def _key(tool: str, args: dict) -> str:
        """计算工具结果缓存键；faq_search 只按 query 缓存（top_k 变化用同一结果截断）"""
        if tool == "faq_search":  # 检索工具特例
            # 检索只按 query 缓存，top_k 变化用同结果截断
            core = {"query": args.get("query", "")}  # 只把 query 纳入键
        else:
            core = args  # 其他工具按完整参数生成键
        return hashlib.md5(  # MD5(工具名 + 序列化参数) 作为缓存键
            f"{tool}:{json.dumps(core, ensure_ascii=False, sort_keys=True, default=str)}"
            .encode("utf-8")).hexdigest()

    def get(self, tool: str, args: dict) -> Any | None:
        """按工具与参数查询缓存，命中且未过期则返回结果，否则返回 None"""
        key = self._key(tool, args)  # 计算缓存键
        with self._lock:  # 加锁读取
            ent = self._entries.get(key)  # 查键
            if ent and time.time() - ent["ts"] < self.ttl:  # 存在且未过期
                self._hits += 1  # 命中计数
                return ent["result"]  # 返回缓存结果
            self._misses += 1  # 未命中计数
            return None  # 调用方重新执行工具

    def put(self, tool: str, args: dict, result: Any):
        """写入工具结果缓存，超容量时淘汰最旧一半（LRU-by-time）"""
        key = self._key(tool, args)  # 计算缓存键
        with self._lock:  # 加锁写入
            self._entries[key] = {"ts": time.time(), "result": result,
                                  "tool": tool}  # 存时间戳/结果/工具名
            if len(self._entries) > CACHE_MAX:  # 超容量
                items = sorted(self._entries.items(), key=lambda kv: kv[1]["ts"])  # 按时间升序
                for k, _ in items[:CACHE_MAX // 2]:  # 最旧一半
                    self._entries.pop(k, None)  # 移除

    def stats(self) -> dict:
        """返回工具缓存命中率统计（命中/未命中/条目数）"""
        total = self._hits + self._misses or 1  # 防除零
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": round(self._hits / total, 3),
                "entries": len(self._entries)}  # 返回统计字典


# 全局单例：引擎与 serve 共用
question_cache = SemanticCache()  # 问题级缓存单例（ReAct 主循环直接 import 使用）
tool_cache = ToolCache()  # 工具级缓存单例


# ═══ 功能块：汇总统计 —— 供 /cache/stats 端点输出两级缓存的命中率 ═══
def cache_stats() -> dict:
    """汇总问题级与工具级缓存统计，供 /cache/stats 端点输出"""
    return {"question": question_cache.stats(), "tool": tool_cache.stats()}  # 两级统计合并返回

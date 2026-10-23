"""
混合检索（Hybrid Search）：BM25 稀疏 + BGE Dense 双向检索 + 轻量 Reranker —— 血压计客服

设计动机（升级自纯 FAISS + 单向量语义检索）：
  1. 型号（HEM-7156）、错误码（E1/E2）等专有名词在纯语义向量里容易被"平均"掉，
     BM25 稀疏检索对精确 token 召回更强，两者互补
  2. RRF（Reciprocal Rank Fusion）融合双路排序，无需调权重即可稳定合并
  3. 轻量 Reranker：词面重叠 + 专有名词加权精排，不引入线上大模型开销
    （如环境装有 cross-encoder 模型，可设 BP_RERANK_MODEL 切换到重排模型）

使用方式：
  from hybrid_search import HybridRetriever
  retriever = HybridRetriever(docs)          # docs: [{id, topic, content, type}]
  hits = retriever.search("HEM-7156 袖带范围", top_k=5)
  print(retriever.stats())                    # 可观测性：dense/sparse 命中贡献
"""

# ═══ 导入区：正则分词 / 数学与向量计算 / 本地模型发现 ═══
import os  # 读 BP_EMBED_PROVIDER / BP_RERANK_MODEL 配置
import re  # 分词与专有名词抽取正则
import math  # BM25 的对数 IDF 与 sigmoid
import logging  # 索引构建/降级日志
from pathlib import Path  # 探测本地重排模型目录
from typing import Any  # 类型标注

import numpy as np  # 向量计算与矩阵操作

logger = logging.getLogger(__name__)  # 本模块日志器

# ── 分词：专有名词（型号/错误码）整词 + 中文 char 1/2-gram ─────────────────────
_TOKEN_WORD = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")  # 英文/数字串（含连字符，如 HEM-7156、E1、KD-828）
_TOKEN_CJK  = re.compile(r"[\u4e00-\u9fff]")  # 单个中文字符


# ═══ 功能块：混合分词器 —— 英文整词 + 中文 1-gram/2-gram 双粒度 ═══
def tokenize(text: str) -> list[str]:
    """混合分词：保留型号（HEM-7156）、错误码（E1）整词；中文按 1-gram+2-gram"""
    tokens: list[str] = []  # 输出 token 列表
    for m in _TOKEN_WORD.finditer(text):  # 找出全部英文/数字串
        tokens.append(m.group(0).lower())  # 整词保留并转小写（BM25 精确匹配专有名词）
    cjk = "".join(_TOKEN_CJK.findall(text))  # 抽出连续中文字符
    tokens.extend(list(cjk))                    # 1-gram：单字兜底
    for i in range(len(cjk) - 1):               # 2-gram：中文语义最小单元
        tokens.append(cjk[i:i + 2])
    return tokens  # 返回混合 token 列表


# ── BM25 稀疏检索 ──────────────────────────────────────────────────────────────

# ═══ 功能块：BM25 稀疏检索器 —— 自研实现避免外部依赖，支持中文 char-gram ═══
class BM25:
    """自研 BM25（避免引入 rank_bm25 依赖），支持中文 char-gram 分词"""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        """构建 BM25 索引。

        :param corpus: 文档文本列表
        :param k1: 词频饱和参数（值越大，词频对得分的贡献越不饱和）
        :param b: 文档长度归一化参数（0=不归一化，1=完全按长度归一化）
        """
        self.k1, self.b = k1, b  # 保存 BM25 两参数
        tokenized = [tokenize(t) for t in corpus]  # 预分词全部文档
        self.doc_len = np.array([len(t) or 1 for t in tokenized], dtype="float32")  # 每篇文档长度（空文档记 1 防除零）
        self.avgdl = float(self.doc_len.mean())  # 平均文档长度（长度归一化基准）
        df: dict[str, int] = {}  # 词 → 出现该词的文档数（document frequency）
        for toks in tokenized:  # 统计 df
            for term in set(toks):  # 每篇文档去重后计一次
                df[term] = df.get(term, 0) + 1
        n = len(corpus) or 1  # 文档总数（空语料记 1）
        # 索引优化：term -> (df, {doc_idx: tf})
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}  # 平滑 IDF：高频词权重低、稀有词权重高
        self.tf_map = {t: {} for t in df}  # 词 → {文档下标: 词频} 倒排结构
        for i, toks in enumerate(tokenized):  # 统计每篇文档内各词频
            for term in toks:
                self.tf_map[term][i] = self.tf_map[term].get(i, 0) + 1
        self.n_docs = n  # 保存文档总数

    def scores(self, query: str) -> np.ndarray:
        """计算查询与每篇文档的 BM25 相关性得分，返回长度等于文档数的向量"""
        scores = np.zeros(self.n_docs, dtype="float32")  # 每篇文档的累计得分清零
        for term in tokenize(query):  # 查询分词后逐词累加
            idf = self.idf.get(term, 0.0)  # 语料外的词 IDF 为 0
            if idf == 0.0:  # 无权重
                continue  # 跳过
            for i, tf in self.tf_map.get(term, {}).items():  # 遍历含该词的文档
                dl = self.doc_len[i]  # 该文档长度
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)  # BM25 分母：长度归一化的词频饱和项
                scores[i] += idf * tf * (self.k1 + 1) / denom  # 累加该词对该文档的得分
        return scores  # 返回全部文档的得分向量


# ── Dense 检索（复用 tools 的 BGE/DashScope 嵌入）───────────────────────────────

# ═══ 功能块：Dense 向量构建 —— 按配置选 DashScope / 本地 BGE / TF 降级 ═══
def _dense_vectors(texts: list[str]) -> np.ndarray:
    """惰性构建/复用向量。优先本地 BGE，其次 DashScope，最后字符 TF 降级。"""
    from tools import _load_st_model, _embed_texts_dashscope, _build_tfidf_vectors  # 复用 tools 的嵌入设施
    if os.getenv("BP_EMBED_PROVIDER", "local") == "dashscope":  # 云端嵌入模式
        vecs = _embed_texts_dashscope(texts)  # 调 API 批量编码
    else:  # 本地模式
        model = _load_st_model()  # 惰性加载本地 BGE 模型
        vecs = (model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                if model is not None else _build_tfidf_vectors(texts))  # 模型可用走 BGE，否则 TF 降级
    return np.array(vecs, dtype="float32")  # 统一转 numpy 矩阵


# ═══ 功能块：查询向量编码 —— 失败返回 None 由上游降级为仅 BM25 ═══
def _embed_query_vec(text: str) -> np.ndarray | None:
    """对查询编码成向量；编码失败时返回 None（下游据此降级为仅 BM25 召回）"""
    from tools import _embed_query  # 复用统一查询编码入口
    vec = _embed_query(text)  # 编码
    return vec if vec is not None else None  # 透传 None 作为降级信号


# ═══ 功能块：中文 2-gram 集合 —— 词面重排的相关性信号 ═══
def _cjk_grams(text: str) -> set[str]:
    """中文 2-gram 集合（1-gram 单字重叠噪音大，不作为相关性信号）"""
    cjk = "".join(_TOKEN_CJK.findall(text))  # 纯中文字符串
    return {cjk[i:i + 2] for i in range(len(cjk) - 1)}  # 相邻两字组成的 2-gram 集合


# ═══ 功能块：重排模型路径发现 —— 环境变量优先，其次探测常见本地目录 ═══
def _default_rerank_model() -> str:
    """重排模型路径：环境变量 BP_RERANK_MODEL 优先，否则自动发现本地已下载模型"""
    env = os.getenv("BP_RERANK_MODEL", "")  # 读环境变量
    if env:  # 显式配置了
        return env  # 直接用
    for cand in ("models/bge-reranker-v2-m3", "models/bge-reranker-base",
                 "bge-reranker-v2-m3", "bge-reranker-base"):  # 探测常见模型目录
        if Path(cand).is_dir():  # 目录存在
            return cand  # 用第一个命中的
    return ""  # 无模型 → 词面打分


# ── 轻量 Reranker ─────────────────────────────────────────────────────────────

# ═══ 功能块：重排器 —— 优先 CrossEncoder 语义精排，无模型降级词面打分 ═══
class LexicalReranker:
    """重排器：默认词面 2-gram + 专有名词加权；检测到本地 CrossEncoder 时自动启用。

    可显式指定：BP_RERANK_MODEL=/path/to/cross-encoder（无本地模型时降级词面打分）。
    """

    def __init__(self):
        """尝试加载 CrossEncoder 重排模型；未配置或加载失败则降级为词面打分"""
        self._model = None  # CrossEncoder 模型槽位
        model_path = _default_rerank_model()  # 探测模型路径
        if model_path:  # 找到候选路径
            try:  # 加载可能失败（依赖缺失等）
                from sentence_transformers import CrossEncoder  # 交叉编码器：对 (查询,文档) 对打分
                self._model = CrossEncoder(model_path)  # 加载
                logger.info(f"已加载重排模型: {model_path}")  # 成功日志
            except Exception as e:  # 加载失败降级到词面打分
                logger.warning(f"重排模型加载失败({e})，降级词面 Rerank")
                self._model = None  # 保持 None

    @property
    def using_cross_encoder(self) -> bool:
        """是否启用了 CrossEncoder（决定精排候选裁剪与打分方式）"""
        return self._model is not None  # 模型加载成功即启用

    def score(self, query: str, text: str) -> float:
        """对单条 (查询, 文档) 打分：CrossEncoder 经 sigmoid 归一化到 (0,1)，否则词面 2-gram + 专有名词加权"""
        if self._model is not None:  # 有 CrossEncoder
            pred = self._model.predict([(query, text)])[0]  # 前向推理取 logits
            # sigmoid 归一化到 (0,1)，与词面分同尺度
            return 1.0 / (1.0 + math.exp(-float(pred)))  # sigmoid(z)=1/(1+e^-z)
        q_grams = _cjk_grams(query)  # 查询中文 2-gram
        d_grams = _cjk_grams(text)  # 文档中文 2-gram
        if not q_grams:  # 查询无中文可比对
            return 0.0  # 零分
        overlap2 = len(q_grams & d_grams) / len(q_grams)  # 2-gram 重叠率（0~1）
        # 专有名词整词（型号/错误码）命中权重更高
        q_words = set(t.lower() for t in _TOKEN_WORD.findall(query))  # 查询英文/数字词集
        d_words = set(t.lower() for t in _TOKEN_WORD.findall(text))  # 文档英文/数字词集
        word_hits = len(q_words & d_words)  # 专有名词命中数
        return overlap2 + 1.0 * word_hits  # 词面分 = 2-gram 重叠率 + 专有名词命中加成

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """批量打分：CrossEncoder 一次前向批量推理，词面模式逐条计算"""
        if self._model is not None:  # CrossEncoder 模式
            preds = self._model.predict(pairs)  # 批量前向
            return [1.0 / (1.0 + math.exp(-float(p))) for p in preds]  # 逐个 sigmoid
        return [self.score(q, t) for q, t in pairs]  # 词面模式逐条打分


# ── 混合检索器 ─────────────────────────────────────────────────────────────────

# ═══ 功能块：混合检索器主类 —— 双路召回 → RRF 融合 → 粗排筛选 → 精排输出 ═══
class HybridRetriever:
    """BM25 + BGE 双路召回 → RRF 融合 → Reranker 精排"""

    def __init__(self, docs: list[dict], rrf_k: int = 60, rerank_ratio: int = 2):
        """构建 BM25 + Dense 双路索引并初始化重排器。

        :param docs: 文档列表，元素含 id/topic/content/type
        :param rrf_k: RRF 融合常数（越大排名差异被压缩得越平缓，默认 60）
        :param rerank_ratio: 双路召回相对 top_k 的放大倍数，保证精排候选充足
        """
        self.docs = docs  # 原始文档列表
        self.texts = [f"{d.get('topic', '')}。{d.get('content', '')}" for d in docs]  # 拼接"主题。正文"作为可检索文本
        self.rrf_k = rrf_k  # RRF 常数
        self.rerank_ratio = rerank_ratio  # 召回放大倍数
        self.bm25 = BM25(self.texts)  # 稀疏路：BM25 索引

        import faiss  # Dense 路：FAISS 内积索引
        vecs = _dense_vectors(self.texts)  # 语料向量化（BGE/降级 TF）
        self._dense_matrix = np.asarray(vecs, dtype="float32")  # 保存向量矩阵供余弦精排复用
        self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])  # 内积索引（向量已归一化 → 内积即余弦）
        self.faiss_index.add(vecs)  # 入库全部语料向量
        logger.info(f"混合检索索引构建完成: {len(docs)} 条文档, dense 维度 {vecs.shape[1]}")  # 构建完成日志

        self._reranker = LexicalReranker()  # 初始化重排器（可能为 CrossEncoder 或词面）
        self._stats: dict[str, Any] = {"searches": 0, "dense_only_hits": 0,
                                        "sparse_only_hits": 0, "joint_hits": 0}  # 可观测性计数器

    def _dense_search(self, vec: np.ndarray, k: int):
        """在 FAISS 内积索引上检索与查询向量最相似的 k 个文档"""
        return self.faiss_index.search(np.asarray(vec, dtype="float32").reshape(1, -1), k)  # 返回 (距离, 下标) 两个矩阵

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """返回 [{id, topic, content, type, score, dense_rank, sparse_rank, method}]"""
        self._stats["searches"] += 1  # 搜索次数计数
        n = len(self.docs)  # 语料总数
        k = min(max(top_k * self.rerank_ratio, top_k), n)  # 实际召回量 = top_k × 放大倍数（不超语料数）

        # 双路召回
        dense_rank: dict[int, int] = {}  # dense 路结果：文档下标 → 名次
        sparse_rank: dict[int, int] = {}  # sparse 路结果：文档下标 → 名次
        vec = _embed_query_vec(query)  # 查询向量化（失败为 None）
        if vec is not None and self.faiss_index.d == vec.shape[1]:  # 有向量且维度匹配才走 dense
            _, indices = self._dense_search(vec, k)  # FAISS 检索
            for rank, idx in enumerate(indices[0], 1):  # 按名次登记
                if 0 <= idx < n and idx not in dense_rank:  # 有效下标且未重复
                    dense_rank[idx] = rank
        bm25_scores = self.bm25.scores(query)  # BM25 打分
        order = np.argsort(-bm25_scores)[:k]  # 得分降序取前 k
        for rank, idx in enumerate(order, 1):  # 按名次登记
            idx = int(idx)  # numpy 下标转 int
            if bm25_scores[idx] > 0 and idx not in sparse_rank:  # 零分不收，避免全 0 噪声
                sparse_rank[idx] = rank

        # RRF 融合
        fused: dict[int, float] = {}  # 文档下标 → RRF 融合分
        for idx, rank in list(dense_rank.items()) + list(sparse_rank.items()):  # 两路结果逐条融合
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)  # RRF：每名次贡献 1/(k+rank)，双路命中累加

        # Reranker 精排：粗排（RRF+dense 余弦）先筛少量候选，再交叉编码器/词面精排
        # （纯词面重叠对"血压计/袖带"等泛词虚高，会淹没语义正确的 FAQ 条目）
        vq = np.asarray(vec, dtype="float32").reshape(-1) if vec is not None else None  # 查询向量摊平备用
        pre_scored: list[tuple[float, int]] = []  # 粗排得分表 (粗分, 文档下标)
        for idx, rrf in fused.items():  # 遍历融合候选
            cos = float(self._dense_matrix[idx] @ vq) if vq is not None else 0.0  # 查询与文档向量的余弦（无向量记 0）
            pre_scored.append((rrf + 0.6 * cos, idx))  # 粗分 = RRF + 0.6×余弦
        pre_scored.sort(key=lambda x: x[0], reverse=True)  # 粗排降序

        use_ce = self._reranker.using_cross_encoder  # 是否有 CrossEncoder
        candidates = pre_scored[:max(top_k * 2, 4)] if use_ce else pre_scored  # CE 精排代价高 → 先裁剪候选；词面便宜全量
        if use_ce:  # CrossEncoder 精排分支
            pairs = [(query, self.texts[idx]) for _, idx in candidates]  # 构造 (查询,文档) 对
            ce_scores = self._reranker.score_batch(pairs)  # 批量精排打分
            scored = [(base + 0.8 * ce, idx)
                      for (base, idx), ce in zip(candidates, ce_scores)]  # 最终分 = 粗分 + 0.8×精排分
        else:  # 词面精排分支
            scored = [(base + 0.15 * self._reranker.score(query, self.texts[idx]), idx)
                      for base, idx in candidates]  # 词面分权重较低，避免泛词虚高
        scored.sort(key=lambda x: x[0], reverse=True)  # 最终排序

        # 命中贡献可观测性
        top_dense = set(dense_rank)  # dense 路命中的文档集合
        top_sparse = set(sparse_rank)  # sparse 路命中的文档集合
        for _, idx in scored[:top_k]:  # 统计最终 top_k 的命中来源
            if idx in top_dense and idx in top_sparse:  # 双路共同命中
                self._stats["joint_hits"] += 1
            elif idx in top_dense:  # 仅 dense
                self._stats["dense_only_hits"] += 1
            elif idx in top_sparse:  # 仅 sparse
                self._stats["sparse_only_hits"] += 1

        results = []  # 最终输出列表
        for score, idx in scored[:top_k]:  # 取前 top_k 组装
            d = self.docs[idx]  # 原始文档
            results.append({
                **d,  # 展开 id/topic/content/type
                "score": round(float(score), 4),  # 最终融合分
                "dense_rank": dense_rank.get(idx),  # dense 名次（未命中为 None）
                "sparse_rank": sparse_rank.get(idx),  # sparse 名次（未命中为 None）
                "method": "hybrid",  # 检索方法标记
            })
        return results  # 返回带完整溯源信息的结果

    def stats(self) -> dict:
        """返回检索可观测性统计：各命中通道（dense/sparse/joint）占比"""
        s = dict(self._stats)  # 拷贝计数器
        total = s["searches"] or 1  # 防除零
        s["hit_distribution"] = {  # 三类命中占比
            "dense_only": round(s["dense_only_hits"] / total, 3),
            "sparse_only": round(s["sparse_only_hits"] / total, 3),
            "joint": round(s["joint_hits"] / total, 3),
        }
        return s  # 返回统计

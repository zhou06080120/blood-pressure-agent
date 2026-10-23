"""hybrid_search 测试：混合分词 / BM25 稀疏检索 / RRF 融合检索器 / 词面重排

测试环境无 BGE 模型（conftest 强制降级）：
  - 索引侧：TF 字符向量降级 → FAISS IndexFlatIP 正常构建
  - 查询侧：无查询向量 → 仅 BM25 召回（确定性）
"""

import numpy as np

from hybrid_search import (
    BM25,
    HybridRetriever,
    LexicalReranker,
    _cjk_grams,
    tokenize,
)


# ═══ 混合分词器 ═════════════════════════════════════════════════════════════

class TestTokenize:
    def test_model_number_kept_whole(self):
        tokens = tokenize("欧姆龙HEM-7156血压计")
        assert "hem-7156" in tokens          # 型号整词保留（小写）

    def test_error_code_whole(self):
        assert "e1" in tokenize("错误代码E1表示什么")

    def test_cjk_unigram_and_bigram(self):
        tokens = tokenize("袖带")
        assert "袖" in tokens and "带" in tokens   # 1-gram
        assert "袖带" in tokens                     # 2-gram

    def test_cjk_grams(self):
        assert _cjk_grams("袖带绑法") == {"袖带", "带绑", "绑法"}


# ═══ BM25 稀疏检索 ═══════════════════════════════════════════════════════════

class TestBM25:
    CORPUS = [
        "欧姆龙HEM-7156 上臂式电子血压计 袖带范围 22-32cm",
        "鱼跃YE660A 上臂式电子血压计 双用户180组",
        "测量前准备：安静环境静坐5分钟，排空膀胱",
    ]

    def test_exact_term_ranks_first(self):
        bm25 = BM25(self.CORPUS)
        scores = bm25.scores("HEM-7156 袖带范围")
        assert int(np.argmax(scores)) == 0   # 型号整词精确命中文档0

    def test_unknow_terms_zero_score(self):
        bm25 = BM25(self.CORPUS)
        assert float(bm25.scores("完全无关词汇").sum()) == 0.0

    def test_empty_corpus_no_crash(self):
        bm25 = BM25([])
        scores = bm25.scores("任何词")  # 空语料不抛异常（n 记 1 防除零）
        assert all(s == 0 for s in scores)


# ═══ 词面重排器（无 CrossEncoder → 词面模式） ════════════════════════════════

class TestLexicalReranker:
    def test_word_mode_not_cross_encoder(self):
        r = LexicalReranker()
        assert r.using_cross_encoder is False  # 测试环境无重排模型

    def test_score_positive_on_overlap(self):
        r = LexicalReranker()
        assert r.score("袖带怎么绑", "袖带下缘距肘窝2-3厘米") > 0

    def test_score_zero_without_cjk(self):
        r = LexicalReranker()
        assert r.score("HEM-7156", "HEM-7156") == 0.0  # 查询无中文 → 0

    def test_proper_noun_bonus(self):
        r = LexicalReranker()
        base = r.score("欧姆龙袖带", "欧姆龙HEM-7156的袖带")
        no_noun = r.score("欧姆龙袖带", "鱼跃血压计的袖带")
        assert base > no_noun  # 专有名词命中加权

    def test_score_batch_consistent(self):
        r = LexicalReranker()
        pairs = [("袖带", "袖带绑法"), ("电池", "电池更换")]
        batch = r.score_batch(pairs)
        assert batch == [r.score(q, d) for q, d in pairs]


# ═══ 混合检索器（TF 降级向量 + 仅 BM25 查询路） ══════════════════════════════

DOCS = [
    {"id": "PROD-HEM-7156", "topic": "欧姆龙HEM-7156",
     "content": "产品代码 HEM-7156，上臂式电子血压计，袖带范围22-32cm，存储21次", "type": "product"},
    {"id": "F002", "topic": "袖带绑法",
     "content": "袖带缠绕上臂距肘窝2-3厘米，松紧以能伸入1-2根手指为宜", "type": "faq"},
    {"id": "TS-漏气", "topic": "故障：漏气",
     "content": "可能原因：袖带气囊破损；解决方案：更换原装袖带", "type": "troubleshoot"},
]


class TestHybridRetriever:
    def test_build_index_with_tf_fallback(self):
        r = HybridRetriever(DOCS)
        assert r.faiss_index.ntotal == 3          # 全部文档向量入库
        assert r._dense_matrix.shape[0] == 3

    def test_search_by_model_number(self):
        r = HybridRetriever(DOCS)
        hits = r.search("HEM-7156 袖带范围", top_k=2)
        assert hits[0]["id"] == "PROD-HEM-7156"   # BM25 整词精确召回
        assert hits[0]["method"] == "hybrid"

    def test_search_result_fields(self):
        r = HybridRetriever(DOCS)
        hits = r.search("袖带怎么绑", top_k=2)
        for h in hits:
            assert {"id", "topic", "content", "type", "score",
                    "dense_rank", "sparse_rank", "method"} <= set(h)

    def test_stats_counts(self):
        r = HybridRetriever(DOCS)
        r.search("袖带", top_k=2)
        r.search("电池", top_k=2)
        s = r.stats()
        assert s["searches"] == 2
        assert "hit_distribution" in s
        # 查询无向量 → 命中只能来自 sparse 路
        assert s["sparse_only_hits"] + s["joint_hits"] > 0

"""
BM25 词面检索门禁 —— CI 可跑的确定性评估基线（无模型权重依赖）

设计动机（P2 评估门禁）：
  1. 混合检索（BM25 + BGE Dense + RRF + Rerank）的完整评测需要 BGE 模型权重，
     CI 环境不装 torch/sentence-transformers —— 因此门禁只锁 BM25 稀疏路：
     纯 numpy 确定性计算，任何环境跑出的数字完全一致
  2. BM25 是混合检索的召回下限：若词面检索本身劣化（分词/索引/语料变更），
     混合检索质量必然受累 —— 把 BM25-only Top1/Top3 命中率锁进 CI，
     检索层重构（如语料改写、分词器调整）引起的回归在合入前即被拦截
  3. 语料与线上一致：直接 from tools import _build_hybrid_docs（20 FAQ +
     6 产品 + 9 故障，35 篇），对 topic + " " + content 拼接文本建 BM25 索引

评测口径：
  - 评测集 evals/evalset.json：30+ 条真实客户口吻中文问句 → 期望命中文档 id
    （id 从 _build_hybrid_docs() 实际输出核对），覆盖 FAQ 咨询 / 产品型号 /
    故障症状三类，难度混合（直述式 + 口语化）
  - 指标：BM25-only Top1 / Top3 命中率（np.argsort 降序取前 3 比对期望 id）
  - 阈值 evals/thresholds.json：实测数字留 3-5 个百分点余量向下取整写入；
    gate 断言不低于阈值，任一 FAIL 以退出码 1 结束（CI job 挂红）

本地运行：
  python evals/gate.py
"""

import os  # 环境变量（KMP 冲突防御必须先于 numpy 设置）

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows 上 faiss/numpy 的 OpenMP 重复加载防御（与 serve.py 同款，必须在 import numpy 之前）

import json  # 评测集与阈值文件读写
import sys  # 注入 src 目录到模块搜索路径
from pathlib import Path  # 定位 evals 同级文件与 src 目录

# 平铺模块架构：把 src 加入 sys.path，使 tools / hybrid_search 可直接导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # BM25 得分向量的 argsort 排序（在 KMP 防御之后导入）

from hybrid_search import BM25  # BM25 稀疏检索器（顶层仅 numpy 依赖，不触发 faiss/BGE）
from tools import _build_hybrid_docs  # 三源语料构建（20 FAQ + 6 产品 + 9 故障）

# 评测资产路径：与 gate.py 同目录
EVALSET_PATH = Path(__file__).parent / "evalset.json"  # 黄金评测集
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"  # 命中率阈值


def load_evalset() -> list[dict]:
    """读取黄金评测集：[{"query": "...", "expect_id": "..."}, ...]"""
    items = json.loads(EVALSET_PATH.read_text(encoding="utf-8"))  # UTF-8 读入
    assert items and all(i.get("query") and i.get("expect_id") for i in items), "评测集条目缺 query/expect_id"  # 结构校验
    return items  # 返回条目列表


def build_index() -> tuple[BM25, list[dict]]:
    """用与线上一致的语料构建 BM25 索引，返回 (索引, 文档列表)。"""
    docs = _build_hybrid_docs()  # 35 篇：id/type/topic/content 四字段
    texts = [f"{d['topic']} {d['content']}" for d in docs]  # topic + 空格 + content 拼接为可检索文本
    return BM25(texts), docs  # BM25 索引与原文档映射（下标 → 文档）


def run_eval() -> dict:
    """执行评测：逐条检索并统计 Top1 / Top3 命中率，返回结果摘要。"""
    items = load_evalset()  # 黄金集
    bm25, docs = build_index()  # 索引
    id_by_idx = [d["id"] for d in docs]  # 下标 → 文档 id 映射

    top1_hits = 0  # Top1 命中数
    top3_hits = 0  # Top3 命中数
    misses = []  # 未命中明细（Top1 未中即记录，便于排查）
    for item in items:  # 逐条评测
        scores = bm25.scores(item["query"])  # BM25 对全部 35 篇打分
        order = np.argsort(-scores)[:3]  # 得分降序取前 3（argsort 默认升序，取负降序）
        top_ids = [id_by_idx[int(i)] for i in order]  # 前三名文档 id
        hit1 = top_ids[0] == item["expect_id"]  # Top1 是否命中
        hit3 = item["expect_id"] in top_ids  # Top3 是否命中
        top1_hits += hit1  # 累计 Top1
        top3_hits += hit3  # 累计 Top3
        if not hit1:  # Top1 未命中：记录明细（Top3 兜中的也记，便于观察边界）
            misses.append({"query": item["query"], "expect": item["expect_id"],
                           "got": top_ids[:3], "in_top3": hit3})  # 实际前三名与是否兜中

    n = len(items)  # 总题数
    return {  # 结果摘要
        "total": n,  # 题数
        "top1_hits": top1_hits,  # Top1 命中数
        "top3_hits": top3_hits,  # Top3 命中数
        "top1_rate": top1_hits / n,  # Top1 命中率
        "top3_rate": top3_hits / n,  # Top3 命中率
        "misses": misses,  # 未命中明细
    }


def main() -> int:
    """门禁入口：跑评测 → 对照阈值 → 逐项 PASS/FAIL，任一 FAIL 返回退出码 1。"""
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))  # 阈值
    result = run_eval()  # 实测

    print("=" * 62)  # 输出区头部
    print(f"BM25 检索门禁：共 {result['total']} 题（evals/evalset.json）")  # 概况
    checks = [  # 两项门禁检查
        ("top1", result["top1_rate"], thresholds["top1"]),  # Top1 命中率
        ("top3", result["top3_rate"], thresholds["top3"]),  # Top3 命中率
    ]
    failed = False  # 是否有 FAIL
    for name, actual, floor in checks:  # 逐项判定
        ok = actual >= floor  # 实测不低于阈值即 PASS
        failed = failed or not ok  # 任一 FAIL 即整体失败
        print(f"  {'PASS' if ok else 'FAIL'}  {name}_hit_rate: 实测 {actual:.1%}"
              f"（{int(round(actual * result['total']))}/{result['total']}）"
              f"  阈值 {floor:.0%}")  # 每项 PASS/FAIL + 实测数字 + 阈值
    if result["misses"]:  # Top1 未命中明细（透明可排查，不影响判定口径）
        print(f"  Top1 未命中明细（{len(result['misses'])} 条）：")  # 明细标题
        for m in result["misses"]:  # 逐条输出
            note = "（Top3 已兜中）" if m["in_top3"] else "（Top3 也未中）"  # 兜中状态
            print(f"    - 期望 {m['expect']} | 实际前三 {m['got']} {note}")  # 明细行
            print(f"      query: {m['query']}")  # 原始问句
    print("=" * 62)  # 输出区收尾
    if failed:  # 有 FAIL
        print("门禁未通过：BM25 检索命中率低于阈值")  # 失败结论
        return 1  # 退出码 1（CI job 挂红）
    print("门禁通过：BM25 检索基线达标")  # 成功结论
    return 0  # 退出码 0


if __name__ == "__main__":  # 直接运行入口
    sys.exit(main())  # 以门禁判定结果作为进程退出码

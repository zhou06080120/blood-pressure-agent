"""
RAGAS / TruLens 风格自动化评估指标 —— 血压计客服 Benchmark

原生实现（无外部依赖，离线可跑）：
  · answer_correctness   答案正确性：ground_truth 关键事实覆盖率（数值 ±容差 匹配 + 字面覆盖）
  · context_relevance    检索相关性：预期工具 vs 实际调用工具的 P/R/F1（含冗余调用惩罚）
  · faithfulness         忠实度：答案中的数值/型号是否在检索上下文中出现（防幻觉指标）

可选升级（装了官方库才生效）：
  · 设置 BP_USE_RAGAS=1 且 pip install ragas → 用官方 ragas 打分（需 LLM）
  · 设置 BP_USE_TRULENS=1 且 pip install trulens-eval → 用官方 feedback functions

环境变量：
  BP_METRIC_BACKEND   native（默认）| ragas | trulens
"""

# ═══ 导入区：re 做文本抽取，os 读后端配置，logging 记录降级警告 ═══
import os  # 读环境变量 BP_METRIC_BACKEND 选择评估后端
import re  # 正则提取数字/中文/型号等文本特征
import logging  # ragas/trulens 不可用时记录警告
from typing import Any  # 类型标注

logger = logging.getLogger(__name__)  # 本模块日志器

BACKEND = os.getenv("BP_METRIC_BACKEND", "native")  # 评估后端：native 原生 / ragas / trulens

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")  # 数字抽取正则：匹配整数与小数（可带负号）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 中文字符抽取正则：CJK 统一表意文字区间


# ═══ 功能块：文本特征抽取（数字 / 中文 n-gram），供三个指标复用 ═══
def _nums(text: str) -> list[float]:
    """从文本中提取所有数字（含小数），用于事实覆盖的数值容差比对"""
    return [float(m) for m in _NUM_RE.findall(text or "")]  # 全文找数字并转 float（空文本安全返回 []）


def _cjk_bigrams(text: str) -> set[str]:
    """提取文本的中文 2-gram 与单字集合，作为字面覆盖度比较的粒度"""
    cjk = "".join(_CJK_RE.findall(text or ""))  # 抽出全部中文字符拼成纯中文字符串
    return {cjk[i:i + 2] for i in range(len(cjk) - 1)} | set(cjk)  # 2-gram 集合 与 单字集合 取并集


# ═══ 功能块：单条事实命中判定 —— 答案是否覆盖了 ground_truth 中的一条要点 ═══
def _fact_hit(fact: str, answer: str, num_tolerance: float = 1.5) -> bool:
    """单条 ground_truth 事实是否被答案覆盖：
    - 事实含数字 → 要求对应数量级数字在答案中出现（±容差）
    - 文本事实 → 中文 bigram 覆盖率 ≥ 0.6"""
    f_nums, a_nums = _nums(fact), _nums(answer)  # 分别提取事实与答案中的数字
    if f_nums and a_nums:  # 双方都含数字 → 走数值容差比对
        # 至少一个关键数字被覆盖（MAP 等计算题允许四舍五入）
        return any(abs(fn - an) <= num_tolerance for fn in f_nums for an in a_nums)  # 任一对数字差值 ≤ 容差即算命中
    if f_nums:  # 事实有数字但答案无数字 → 未覆盖
        return False
    f_grams = _cjk_bigrams(fact)  # 事实侧的中文 2-gram+单字集合
    if not f_grams:  # 事实里没有中文可比对
        return True  # 视为覆盖（无从判定）
    a_text = "".join(_CJK_RE.findall(answer or ""))  # 答案的纯中文文本
    covered = sum(1 for g in f_grams if g in a_text)  # 统计答案中出现的事实 gram 数
    return covered / len(f_grams) >= 0.6  # 覆盖率 ≥ 0.6 判定事实命中


# ═══ 功能块：指标 1 答案正确性 —— ground_truth 关键事实覆盖率的 P/R/F1 ═══
def answer_correctness(answer: str, ground_truth: list[str]) -> dict:
    """答案正确性：P/R/F1。ground_truth 为关键事实列表。"""
    if not ground_truth:  # 无标注事实
        return {"precision": None, "recall": None, "f1": None,
                "matched": 0, "total": 0, "missing": []}  # 返回全空指标（缺省而非报错）
    matched = [f for f in ground_truth if _fact_hit(f, answer)]  # 命中的事实列表
    missing = [f for f in ground_truth if f not in matched]  # 未命中的事实列表（诊断信息）
    recall = len(matched) / len(ground_truth)  # 召回率 = 命中数 / 标注事实总数
    precision = recall  # 本机近似：事实覆盖率即精确度代理（无标注干扰项时）
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0  # F1 调和平均（分母为 0 防护）
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "matched": len(matched),
            "total": len(ground_truth), "missing": missing}  # 返回完整指标字典


# ═══ 功能块：指标 2 工具调用相关性 —— 预期工具 vs 实际调用的 P/R/F1 与冗余惩罚 ═══
def context_relevance(expected_tools: list[str], tools_used: list[str]) -> dict:
    """检索上下文相关性：工具调用与预期工具的 P/R/F1 + 冗余调用惩罚。

    计数按调用次数而非去重集合：expected 中重复出现表示预期多次调用，
    tools_used 中超过预期的重复（或使用预期外工具）计入冗余惩罚。"""
    exp, used = list(expected_tools), list(tools_used)  # 拷贝入参（避免改动原列表）
    if not exp:  # 无解问题：不应调用任何工具
        redundant = len(used)  # 全部实际调用都算冗余
        precision = 0.0 if used else 1.0  # 调了工具精确度为 0；一次没调为满分 1
        recall = 1.0  # 预期为空天然满足
        return {"precision": precision, "recall": recall,
                "f1": 0.0 if used else 1.0,
                "redundant_calls": redundant, "correct_tools": 0,
                "total_tools_used": len(used)}  # 返回无解题的指标
    correct = sum(min(used.count(t), exp.count(t)) for t in set(exp))  # 每个预期工具按 min(实际,预期) 次数计为正确调用
    redundant = len(used) - correct  # 实际调用总数减正确数 = 冗余调用
    precision = correct / len(used) if used else 0.0  # 精确度 = 正确 / 实际总数
    recall = correct / len(exp) if exp else 1.0  # 召回率 = 正确 / 预期总数
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0  # F1
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "redundant_calls": redundant,
            "correct_tools": correct, "total_tools_used": len(used)}  # 返回常规指标


# ═══ 功能块：指标 3 忠实度 —— 答案中的数值/型号是否能在检索上下文找到（防幻觉） ═══
def faithfulness(answer: str, contexts: list[str]) -> dict:
    """忠实度：答案中的数值/型号是否出现在检索上下文中（防幻觉代理指标）。"""
    if not contexts:  # 没有检索上下文可对照
        return {"faithfulness": None, "hallucinated_numbers": [],
                "note": "无检索上下文"}  # 无法计算，返回 None 并注明原因
    ctx_text = "\n".join(contexts or [])  # 把全部上下文拼成一段长文本
    answer_nums = _nums(answer)  # 答案中的全部数字
    ctx_nums = _nums(ctx_text)  # 上下文中的全部数字
    hallucinated = [n for n in answer_nums  # 找出答案里"上下文中 ±1.5 容差内找不到"的数字 → 疑似幻觉
                    if not any(abs(n - c) <= 1.5 for c in ctx_nums)]
    # 型号等专有名词：答案中出现的型号必须出现在上下文
    models = re.findall(r"[A-Za-z]{2,6}-?\d{3,5}", answer or "")  # 抽取 HEM-7156 / KD-828 这类型号 token
    hallucinated_models = [m for m in models if m not in ctx_text]  # 型号在上下文找不到 → 幻觉型号
    total = len(answer_nums) + len(models)  # 需要核验的实体总数（数字 + 型号）
    bad = len(hallucinated) + len(hallucinated_models)  # 幻觉实体数
    score = round((total - bad) / total, 3) if total else 1.0  # 忠实度 = 1 - 幻觉占比（无可核验项给满分）
    return {"faithfulness": score,
            "hallucinated_numbers": hallucinated,
            "hallucinated_models": hallucinated_models}  # 返回分数与幻觉明细


# ── 官方库可选接入（未安装则回退 native）────────────────────────────────────────

# ═══ 功能块：RAGAS 官方后端适配 —— 可用则返回官方分数，异常则回退 native ═══
def _ragas_scores(answer: str, question: str, contexts: list[str]) -> dict | None:
    try:  # 任何一步失败都回退
        from datasets import Dataset  # 构造 ragas 输入数据集
        from ragas import evaluate  # ragas 评估入口
        from ragas.metrics import faithfulness as rf, answer_relevancy as rar  # 官方忠实度/相关性指标
        ds = Dataset.from_dict([{  # 把单条问答打包成 HuggingFace Dataset
            "question": [question], "answer": [answer],
            "contexts": [contexts or [""]],
        }])
        result = evaluate(ds, metrics=[rf, rar])  # 用 ragas 打分
        scores = result.to_pandas()  # 转成 DataFrame 便于取值
        return {"faithfulness": float(scores["faithfulness"][0]),
                "answer_relevancy": float(scores["answer_relevancy"][0])}  # 取第一行的两个官方分数
    except Exception as e:  # 库未安装 / 需要 LLM key / 运行报错等
        logger.warning(f"ragas 不可用({e})，回退 native 指标")  # 记录警告
        return None  # 返回 None 表示用原生指标


# ═══ 功能块：TruLens 官方后端适配 —— 用 groundedness 反馈函数打分 ═══
def _trulens_scores(answer: str, question: str) -> dict | None:
    try:  # 任何一步失败都回退
        from trulens_eval import Feedback  # TruLens 核心类
        from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI  # OpenAI 供给端
        provider = TruOpenAI()  # 初始化供给端（需 OPENAI_API_KEY）
        grounded = Feedback(provider.groundedness_measure).on_output()  # 构建接地度反馈函数（作用于输出）
        result = grounded(answer, question)  # 对 (答案, 问题) 打分
        return {"groundedness": float(result)}  # 返回接地度分数
    except Exception as e:  # 库未安装等
        logger.warning(f"trulens 不可用({e})，回退 native 指标")  # 记录警告
        return None


# ═══ 功能块：统一评估入口 —— 按入参与后端自动组装各指标并返回汇总字典 ═══
def evaluate_answer(*, question: str, answer: str,
                    ground_truth: list[str] | None = None,
                    expected_tools: list[str] | None = None,
                    tools_used: list[str] | None = None,
                    contexts: list[str] | None = None) -> dict:
    """统一评估入口，自动按 BP_METRIC_BACKEND 选择后端。"""
    metrics: dict[str, Any] = {"backend": BACKEND}  # 结果容器，先记录使用的后端
    if ground_truth is not None:  # 提供了标注事实 → 计算答案正确性
        metrics["answer_correctness"] = answer_correctness(answer, ground_truth)
    if expected_tools is not None and tools_used is not None:  # 同时提供预期与实际工具 → 计算相关性
        metrics["context_relevance"] = context_relevance(expected_tools, tools_used)
    if contexts is not None:  # 提供了检索上下文 → 计算忠实度
        metrics["faithfulness"] = faithfulness(answer, contexts)

    if BACKEND == "ragas":  # 后端选了 ragas → 叠加官方分数
        official = _ragas_scores(answer, question, contexts)
        if official:  # 官方库可用时才写入
            metrics["ragas_official"] = official
    elif BACKEND == "trulens":  # 后端选了 trulens → 叠加官方分数
        official = _trulens_scores(answer, question)
        if official:
            metrics["trulens_official"] = official
    return metrics  # 返回全部指标汇总

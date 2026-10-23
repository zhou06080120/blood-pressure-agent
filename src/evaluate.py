"""
对比评估：手写 Prompt 解析版 vs Function Calling 版 —— 血压计客服

评估维度（对应 react_financial_agent/src/evaluate.py）：
  1. 格式稳定性（手写版专属：成功解析率）
  2. 步骤数（完成同一问题用了几步）
  3. 耗时（秒）
  4. 是否给出 Final Answer（成功率）

使用方式：
  python evaluate.py
  python evaluate.py --output ../evaluation/compare_result.json
"""

# ═══ 导入区：标准库 + 环境准备（项目模块在运行时才延迟导入） ═══
import os  # 设置 KMP 兼容开关、定位 src 目录
import sys  # 把当前目录注入模块搜索路径
import json  # 评估结果序列化写盘
import time  # 计时对比两种实现的耗时
import argparse  # 解析 --output/--max_steps 参数
import logging  # 压制第三方库的冗余日志

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许重复加载 OpenMP，防止 numpy/faiss 崩溃
sys.path.insert(0, os.path.dirname(__file__))  # 把 src/ 加入搜索路径，使下面能 import react_manual 等模块

logging.basicConfig(level=logging.WARNING)  # 全局日志级别设为警告以上

# 评估用问题集：覆盖不同工具组合和难度
EVAL_QUESTIONS = [  # 5 道评估题：覆盖检索/计算/产品查询/多故障/安全边界
    # ── Q1：FAQ 检索 + 故障排查两个工具的组合 ──
    {
        "id": "Q1",  # 题目编号
        "question": "血压计袖带应该怎么绑才准？测量值偏高怎么办？",  # 问题原文
        "expected_tools": ["faq_search", "troubleshoot"],  # 期望调用的工具序列
        "note": "FAQ检索 + 故障排查",  # 出题意图说明
    },
    # ── Q2：两次计算 + 标准对照（calculator ×2 + measurement_guide） ──
    {
        "id": "Q2",
        "question": "我的收缩压是135，舒张压是85，请帮我算一下平均动脉压和脉压差，并判断是否正常？",  # 问题原文
        "expected_tools": ["calculator", "calculator", "measurement_guide"],  # 期望工具链
        "note": "多次计算 + 标准对比",
    },
    # ── Q3：单次产品规格查询 ──
    {
        "id": "Q3",
        "question": "欧姆龙HEM-7156这款血压计的袖带适用范围和供电方式是什么？",  # 问题原文
        "expected_tools": ["product_lookup"],  # 只需一次 product_lookup
        "note": "单次产品查询",
    },
    # ── Q4：两个故障需分次排查 ──
    {
        "id": "Q4",
        "question": "血压计蓝牙连不上APP怎么办？数据不同步怎么解决？",  # 问题原文
        "expected_tools": ["troubleshoot", "troubleshoot"],  # 两个症状各查一次
        "note": "多故障排查",
    },
    # ── Q5：医学越界问题，期望不调任何工具直接安全拒绝 ──
    {
        "id": "Q5",
        "question": "请帮我诊断我是否患有高血压，应该吃什么药？",  # 问题原文
        "expected_tools": [],  # 无解/越界题：不应调用任何工具
        "note": "超出能力边界（医学诊断），应拒绝作答并建议就医",
    },
]


# ═══ 功能块：单题执行器 —— 跑一次 ReAct 并收集步数/耗时/解析错误等指标 ═══
def _run_single(mode: str, question: str, max_steps: int = 10) -> dict:
    """运行单个问题，收集评估指标"""
    if mode == "manual":  # 手写版：正则解析 Prompt 输出
        from react_manual import run as react_run
    else:  # FC 版：由 API 解析 tool_calls
        from react_function_calling import run as react_run

    steps        = []  # 收集每一步事件
    parse_errors = 0  # 解析失败计数（手写版专属指标）
    start        = time.time()  # 计时起点
    final_answer = None  # 最终答案
    success      = False  # 是否成功给出 Final Answer

    for step_data in react_run(question, max_steps=max_steps):  # 消费生成器逐步执行
        steps.append(step_data)  # 记录每步事件
        if step_data["type"] == "unparseable":  # 手写版输出格式解析失败
            parse_errors += 1  # 计数
        if step_data["type"] == "final":  # 拿到最终答案
            final_answer = step_data["answer"]
            success = True  # 标记成功

    elapsed = time.time() - start  # 总耗时（秒）
    action_steps = [s for s in steps if s["type"] == "action"]  # 过滤出工具调用步
    tools_used   = [s["action"] for s in action_steps]  # 按顺序记录实际用过的工具名

    return {  # 返回单题评估结果字典
        "mode":         mode,  # 实现方式 manual/fc
        "total_steps":  len(action_steps),  # 工具调用步数
        "elapsed_s":    round(elapsed, 1),  # 耗时（保留 1 位小数）
        "success":      success,  # 成功率布尔值
        "parse_errors": parse_errors,   # 手写版专属指标
        "tools_used":   tools_used,  # 实际工具调用序列
        "final_answer": (final_answer or "")[:200],  # 答案预览（截断 200 字）
    }


# ═══ 功能块：对比评估主入口 —— 每题在 manual/fc 双模式下运行并汇总对比 ═══
def evaluate(output_path: str | None = None, max_steps: int = 10):
    """对 EVAL_QUESTIONS 在 manual/fc 两模式下逐题运行并汇总对比（步数/耗时/成功率/解析错误）"""
    results = []  # 收集所有 (题, 模式) 组合的结果
    # v2：评测前清空语义缓存，保证两个模式的时延对比公平
    try:  # 缓存模块可能未安装，失败不阻断评测
        from semantic_cache import question_cache, tool_cache  # 导入全局两级缓存单例
        question_cache._entries.clear()  # 清空问题级缓存条目
        tool_cache._entries.clear()  # 清空工具级缓存条目
    except ImportError:  # 缺模块时跳过清缓存
        pass

    for q in EVAL_QUESTIONS:  # 逐题评估
        print(f"\n{'─'*60}")  # 分隔线
        print(f"[{q['id']}] {q['question']}")  # 打印题目
        print(f"  预期工具: {q['expected_tools']}  ({q['note']})")  # 打印预期工具链与出题意图

        for mode in ["manual", "fc"]:  # 两种实现各跑一遍
            print(f"  运行 [{mode}]...", end=" ", flush=True)  # 实时提示当前模式（不换行）
            r = _run_single(mode, q["question"], max_steps)  # 执行单题并收集指标
            print(f"步骤:{r['total_steps']}  耗时:{r['elapsed_s']}s  成功:{r['success']}")  # 打印单题结果
            results.append({**q, **r})  # 题目元信息 + 运行指标合并后追加

    # 汇总对比表
    print(f"\n{'='*60}")  # 表头分隔
    print(f"{'ID':<4} {'Mode':<8} {'Steps':<7} {'Time(s)':<9} {'Success':<8} {'ParseErr'}")  # 列标题
    print(f"{'─'*60}")  # 分隔线
    for r in results:  # 逐行打印明细
        print(
            f"{r['id']:<4} {r['mode']:<8} {r['total_steps']:<7} "  # ID/模式/步数
            f"{r['elapsed_s']:<9} {str(r['success']):<8} {r['parse_errors']}"  # 耗时/成功/解析错误数
        )

    # 按维度汇总
    for mode in ["manual", "fc"]:  # 分别统计两种模式
        mode_results = [r for r in results if r["mode"] == mode]  # 过滤出该模式全部结果
        avg_steps   = sum(r["total_steps"] for r in mode_results) / len(mode_results)  # 平均步数
        avg_time    = sum(r["elapsed_s"]    for r in mode_results) / len(mode_results)  # 平均耗时
        success_rate = sum(r["success"] for r in mode_results) / len(mode_results)  # 成功率
        total_errors = sum(r["parse_errors"] for r in mode_results)  # 解析错误总数
        print(f"\n[{mode}] 平均步数:{avg_steps:.1f}  平均耗时:{avg_time:.1f}s  "  # 汇总打印
              f"成功率:{success_rate:.0%}  解析错误总数:{total_errors}")

    if output_path:  # 指定了输出路径时写盘
        # numpy scalar 安全转换
        def _safe(v):  # 嵌套工具函数
            """将 numpy 标量转换为原生 Python 类型，保证 json.dumps 可序列化"""
            return v.item() if hasattr(v, "item") else v  # 有 item() 就转换，否则原样返回

        safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]  # 逐值做安全转换
        with open(output_path, "w", encoding="utf-8") as f:  # 打开输出文件
            json.dump(safe_results, f, ensure_ascii=False, indent=2)  # 中文不转义、缩进 2 格写盘
        print(f"\n结果已保存至 {output_path}")  # 提示保存位置


if __name__ == "__main__":  # 直接运行本文件才执行评估
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--output",    default=None, help="保存 JSON 结果的路径")  # 结果输出路径
    parser.add_argument("--max_steps", type=int, default=10)  # 最大步数
    args = parser.parse_args()  # 解析命令行
    evaluate(args.output, args.max_steps)  # 执行对比评估

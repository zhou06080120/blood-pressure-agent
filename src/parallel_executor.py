"""
并行工具执行器 —— 血压计客服 Agent

FC 模式下模型一次回复可能携带多个 tool_calls（如同时要 product_lookup + faq_search），
原先按顺序逐个执行，每个回合叠加 LLM 等待 → 累积出 7.7s 级延迟。

本模块用 ThreadPoolExecutor 并发执行同一轮内无依赖的工具调用：
  - 工具均为纯函数/本地检索，无共享可变状态 → 并发安全
  - 结果按原顺序返回，保持 messages 中 tool_call_id 对齐
  - 单例线程池复用，避免每轮建池开销

环境变量：
  BP_PARALLEL_WORKERS 并发线程数，默认 4
"""

# ═══ 导入区：线程池 / 计时 / 日志 / 环境变量 ═══
import os  # 读 BP_PARALLEL_WORKERS 配置并发数
import time  # perf_counter 为每次工具调用计时
import logging  # 记录执行异常
from concurrent.futures import ThreadPoolExecutor  # 线程池：并发执行互不依赖的工具

logger = logging.getLogger(__name__)  # 本模块日志器

WORKERS = int(os.getenv("BP_PARALLEL_WORKERS", "4"))  # 并发线程数，默认 4

_executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="bp-tool")  # 模块级单例线程池：避免每轮重建池的开销


# ═══ 功能块：并行执行入口 —— 同轮多个无依赖工具调用并发跑，结果按提交顺序返回 ═══
def run_parallel(calls: list[dict]) -> list[dict]:
    """并发执行工具调用。

    :param calls: [{"name": str, "fn": Callable, "args": dict, "cache_key_args": dict|None}]
                  cache_key_args 与 args 相同则省略（走语义缓存 key 的调用方自行处理）
    :return: 与原顺序一致的 [{"name", "args", "observation", "error", "elapsed_ms"}]
    """
    if not calls:  # 空调用列表
        return []  # 直接返回空列表
    if len(calls) == 1:  # 只有一个调用
        return [_run_one(calls[0])]  # 无并发收益，直接同步执行

    futures = {_executor.submit(_run_one, c): idx for idx, c in enumerate(calls)}  # 提交全部任务，建立 future → 原始下标 的映射（保证结果顺序）
    results = [None] * len(calls)  # 预分配结果槽位，下标对齐
    for fut in futures:  # 逐个收集各 future
        idx = futures[fut]  # 取该 future 对应的原始下标
        try:
            results[idx] = fut.result()  # 阻塞等待该任务完成并回填到原位置
        except Exception as e:  # 兜底：工具内部异常也返回结构化错误而非崩溃
            c = calls[idx]  # 取出对应调用描述
            results[idx] = {"name": c["name"], "args": c["args"],
                            "observation": f"工具执行异常: {e}", "error": str(e),
                            "elapsed_ms": 0.0}  # 包装为统一错误结构体（耗时记 0）
    return results  # 返回与输入顺序一致的结果列表


# ═══ 功能块：单任务执行器 —— 真正调用工具函数并把一切异常转为结构化结果 ═══
def _run_one(c: dict) -> dict:
    """执行单个工具调用，捕获类型错误与其他异常并包装为结构化结果（不向外抛异常）

    error 字段语义：None=执行成功；非 None=失败描述（调用方据此决定是否写入缓存）
    """
    t0 = time.perf_counter()  # 记录开始时间
    error = None  # 成功标记：仅执行异常时置为错误描述
    try:
        observation = c["fn"](**c["args"])  # 以关键字参数展开方式真正执行工具函数
    except TypeError as e:  # 参数名不匹配等类型错误 → 回给模型让它自行修正参数
        observation = f"工具参数错误: {e}"
        error = str(e)  # 标记失败：此类结果不应进入工具缓存
    except Exception as e:  # 其他运行时异常
        observation = f"工具执行异常: {e}"
        error = str(e)  # 标记失败
    elapsed = round((time.perf_counter() - t0) * 1000, 1)  # 毫秒耗时（保留 1 位小数）
    return {"name": c["name"], "args": c["args"],
            "observation": str(observation), "error": error, "elapsed_ms": elapsed}  # error 如实反映执行结果

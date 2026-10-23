"""
决策状态机 + 幻觉自纠错护栏 —— 血压计客服 Agent

解决传统 ReAct 的两个顽疾：
  1. 多工具交替调用时 LLM 反复调用同一工具（死循环、浪费 latency）
     → ToolCallMonitor 记录调用历史（工具+参数哈希），相同调用超出阈值强制阻断并注入纠错提示
  2. 工具调用失败（参数错误/未知工具/计算报错）时 Agent 直接崩掉或陷入错误循环
     → 失败计数 + 恢复建议注入：连续失败超出阈值自动切换策略（降级到 faq_search 兜底）

同时引入 AgentStateMachine：显式建模 INIT/PLANNING/EXECUTING/OBSERVING/RECOVERING/FINAL，
限制最大循环深度（BP_MAX_LOOP_DEPTH），达到上限强制收敛给出兜底答案而非无限循环。

环境变量：
  BP_MAX_LOOP_DEPTH   最大循环深度，默认 12
  BP_MAX_SAME_CALL    相同工具+参数最大重复次数，默认 2（第 3 次被阻断）
  BP_MAX_CONSEC_ERROR 连续工具失败阈值，默认 2（超过自动切换策略）
"""

# ═══ 导入区：环境变量配置 / JSON+MD5 做调用指纹 / 日志 ═══
import os  # 读 BP_MAX_LOOP_DEPTH 等护栏阈值配置
import json  # 参数序列化（生成调用指纹）
import hashlib  # MD5 哈希生成"工具+参数"的稳定键
import logging  # 非法迁移/阻断等审计日志

logger = logging.getLogger(__name__)  # 本模块日志器

MAX_LOOP_DEPTH     = int(os.getenv("BP_MAX_LOOP_DEPTH", "12"))  # 最大循环深度（防止无限 ReAct 轮次）
MAX_SAME_CALL      = int(os.getenv("BP_MAX_SAME_CALL", "2"))  # 相同(工具+参数)允许的最大重复次数
MAX_CONSEC_ERRORS  = int(os.getenv("BP_MAX_CONSEC_ERROR", "2"))  # 连续失败阈值（超过注入策略调整提示）

# 判定"工具失败"的错误信号（均来自 tools.py 的降级返回约定）
_ERROR_SIGNALS = ("工具参数错误", "未知工具", "计算出错", "执行出错", "加载失败")  # 观察文本含任一信号即视为失败


# ═══ 功能块 1：决策状态机 —— 显式建模 ReAct 各阶段，限制非法迁移与循环深度 ═══
class AgentStateMachine:
    """决策状态机：控制 ReAct 循环的合法状态迁移与循环深度"""

    INIT, PLANNING, EXECUTING, OBSERVING, RECOVERING, FINAL, GIVEUP = (
        "init", "planning", "executing", "observing", "recovering", "final", "giveup")  # 七个状态常量：初始化/规划/执行/观察/恢复/结束/放弃

    _TRANSITIONS = {  # 合法迁移表：当前状态 -> 允许的后继状态集合
        INIT:      {PLANNING},  # 初始只能进入规划
        PLANNING:  {EXECUTING, FINAL},  # 规划后执行或直接给最终答案
        EXECUTING: {OBSERVING, RECOVERING},  # 执行后观察；执行异常可转恢复
        OBSERVING: {PLANNING, FINAL},  # 观察后继续规划或收敛结束
        RECOVERING:{PLANNING, GIVEUP},  # 恢复后重规划或放弃
        FINAL:     set(),  # 终态：不再迁移
        GIVEUP:    set(),  # 放弃态：不再迁移
    }

    def __init__(self, max_depth: int | None = None):
        """初始化状态机。

        :param max_depth: 最大循环深度，None 时回退到环境变量 BP_MAX_LOOP_DEPTH
        """
        self.state = self.INIT  # 当前状态：从 INIT 开始
        self.max_depth = max_depth or MAX_LOOP_DEPTH  # 循环深度上限
        self.depth = 0  # 已消耗的循环深度（每进入 EXECUTING +1）
        self.history: list[str] = []  # 状态迁移历史（调试/可观测）

    def transition(self, target: str) -> bool:
        """尝试将状态迁移到目标状态。

        :param target: 目标状态（须是当前状态在 _TRANSITIONS 中的合法后继）
        :return: 迁移合法则更新内部状态并返回 True，非法迁移忽略并返回 False
        """
        if target not in self._TRANSITIONS.get(self.state, set()):  # 目标不在合法后继集合
            logger.warning(f"非法状态迁移 {self.state} -> {target}，忽略")  # 审计日志
            return False  # 拒绝迁移（状态保持不变）
        self.state = target  # 更新当前状态
        # 仅在进入执行态时累加循环深度：一次"思考→执行→观察"构成一个完整轮次
        if target == self.EXECUTING:  # 进入执行态
            self.depth += 1  # 深度计数
        self.history.append(target)  # 记录迁移轨迹
        return True  # 迁移成功

    @property
    def loop_exhausted(self) -> bool:
        """循环深度是否已达上限（主循环据此判断是否强制收敛、避免死循环）"""
        return self.depth >= self.max_depth  # 已消耗深度 ≥ 上限即耗尽

    def snapshot(self) -> dict:
        """返回状态机快照，供下游步骤事件/日志携带可观测信息"""
        return {"state": self.state, "depth": self.depth,
                "max_depth": self.max_depth, "loop_exhausted": self.loop_exhausted}  # 当前状态+深度+是否耗尽


# ═══ 功能块 2：工具调用监视器 —— 相同调用去重阻断 + 连续失败自纠错 ═══
class ToolCallMonitor:
    """工具调用历史去重 + 失败自纠错"""

    def __init__(self, max_same_call: int | None = None,
                 max_consec_errors: int | None = None):
        """初始化调用监视器。

        :param max_same_call: 相同(工具+参数)允许的最大重复次数（第 N+1 次被阻断）
        :param max_consec_errors: 连续失败阈值，超过后触发策略调整提示
        """
        self.max_same_call = max_same_call or MAX_SAME_CALL  # 相同调用上限（默认 2）
        self.max_consec_errors = max_consec_errors or MAX_CONSEC_ERRORS  # 连续失败阈值（默认 2）
        self.calls: list[dict] = []  # 全部调用历史记录
        self.consec_errors = 0  # 当前连续失败计数
        self.blocked_duplicates = 0  # 累计阻断的重复调用数
        self.recoveries = 0  # 累计触发的恢复次数

    @staticmethod
    def _key(tool: str, args: dict) -> str:
        """生成工具调用的稳定哈希键：参数按序序列化后求 MD5，用于识别完全相同调用"""
        try:  # 参数可能含不可序列化对象
            arg_str = json.dumps(args, ensure_ascii=False, sort_keys=True,
                                 default=str)  # 键排序保证相同参数字典顺序不同也能得到相同键
        except Exception:  # 序列化失败兜底
            arg_str = str(args)
        return hashlib.md5(f"{tool}:{arg_str}".encode("utf-8")).hexdigest()  # MD5("工具名:参数串")

    def check_before(self, tool: str, args: dict) -> tuple[bool, str]:
        """调用工具前检查。返回 (是否阻断, 阻断原因/hint)。

        · 相同(工具+参数)第 max_same_call+1 次调用 → 阻断，杜绝原地打转
        · 工具本身不同但连续调用同一工具的语义参数 → 由原因提示引导
        """
        key = self._key(tool, args)  # 本次调用的指纹键
        same = sum(1 for c in self.calls if c["key"] == key)  # 历史上完全相同调用次数
        if same >= self.max_same_call:  # 已达上限还再来 → 硬阻断
            self.blocked_duplicates += 1  # 阻断计数
            return True, (f"检测到对 {tool} 的完全相同的调用已执行 {same} 次，"
                          f"请勿重复调用：请更换参数或改用其他工具（如 faq_search），"
                          f"若信息已足够请直接给出最终答案。")  # 返回阻断 + 纠错提示（会被回填给模型）
        same_tool = [c for c in self.calls if c["tool"] == tool]  # 同名工具（参数不同）的调用历史
        if len(same_tool) >= self.max_same_call * 2:  # 同工具反复尝试仍无进展 → 软提示
            return False, (f"提示：{tool} 已连续调用 {len(same_tool)} 次仍无进展，"
                           f"建议切换策略（faq_search 兜底或直接基于现有信息作答）。")  # 不阻断但给引导
        return False, ""  # 正常放行

    def record(self, tool: str, args: dict, observation: str):
        """记录一次工具调用及其结果，并据此更新连续失败计数（用于自动纠错）"""
        self.calls.append({"tool": tool, "args": args,
                           "key": self._key(tool, args),
                           "observation": observation[:200]})  # 观察文本截断 200 字防止历史膨胀
        if _is_error(observation):  # 结果是失败信号
            self.consec_errors += 1  # 连续失败 +1
        else:  # 成功
            self.consec_errors = 0  # 连续失败清零

    def recovery_hint(self) -> str | None:
        """连续失败达到阈值 → 返回策略调整提示，并复位计数。"""
        if self.consec_errors >= self.max_consec_errors:  # 连续失败达阈值
            self.recoveries += 1  # 恢复计数
            self.consec_errors = 0  # 复位计数（提示只注入一次）
            return ("连续 {n} 次工具调用失败。请停止重试同一思路：改用 faq_search 检索通用解答，"
                    "或直接用已知医学常识作答并附就医建议。".format(
                        n=self.max_consec_errors))  # 返回策略切换提示文本
        return None  # 未达阈值：无需提示

    def stats(self) -> dict:
        """返回监视器统计信息，用于 /health 等端点的可观测性输出"""
        return {"total_calls": len(self.calls),
                "blocked_duplicates": self.blocked_duplicates,
                "recoveries": self.recoveries,
                "consec_errors": self.consec_errors}  # 调用总数/阻断数/恢复数/当前连败


# ═══ 功能块 3：失败信号判定 —— 依据 tools.py 的降级返回约定 ═══
def _is_error(observation: str) -> bool:
    """判定观察结果是否为工具失败信号（依据 tools.py 的降级返回约定）"""
    return any(sig in observation for sig in _ERROR_SIGNALS)  # 观察文本含任一错误信号即失败


# ── 便捷封装：一次性拿到双护栏 ─────────────────────────────────────────────────

# ═══ 便捷工厂：主循环一行代码同时创建状态机 + 监视器 ═══
def make_guards() -> tuple[AgentStateMachine, ToolCallMonitor]:
    """便捷工厂：一次性创建状态机与调用监视器，供两种 ReAct 主循环复用"""
    return AgentStateMachine(), ToolCallMonitor()  # 返回双护栏元组

"""
故障排查决策图谱（Decision Graph）—— 血压计客服 Agent

将原先扁平的 TROUBLESHOOT_MAP（症状 → 原因/方案列表）升级为带因果关系的决策图：
  - 节点：症状节点（含可能原因与解决方案）、检查项节点、就医建议节点
  - 边：原因 → 向下钻取的子检查 / 相关症状（如"测不出值"的原因"袖带漏气"→ 链接"漏气"排查）
  - 推演：resolve(symptom) 输出按优先级排序的决策路径（先易后难），
    支持多跳联合推理（结构化逻辑 + 语义检索互补：语义负责"像什么"，图谱负责"查什么"）

使用方式：
  from troubleshoot_graph import TROUBLESHOOT_GRAPH, graph_resolve
  path = graph_resolve("测不出值")   # 返回结构化决策路径（节点 + 步骤 + 后续问题）
"""

# ═══ 导入区：os 读推演深度配置，logging 记录降级，re 提取中文 2-gram ═══
import os  # 读 BP_GRAPH_MAX_DEPTH 覆盖默认深度
import re  # 中文 2-gram 提取（交互式 walk 的反馈提及判定）
import logging  # 图谱加载/推演日志

logger = logging.getLogger(__name__)  # 本模块日志器

_TOKEN_CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 中文字符正则（预编译，2-gram 提取用）

# ── 图谱：在 tools.TROUBLESHOOT_MAP 之上补充结构化因果边 ────────────────────────
# 边定义：{父症状: [(父原因关键词, 目标节点, 关系), ...]}
# 目标节点可以是另一症状（深入排查）、检查项或就医建议节点
GRAPH_EDGES: dict[str, list[tuple[str, str, str]]] = {  # 因果边表：让扁平手册升级为可多跳的决策图
    "测不出值": [  # "测不出值"的下游边
        ("漏气", "漏气", "袖带漏气需专项排查"),  # 原因含"漏气"→ 跳转"漏气"症状继续深挖
        ("电池", "无法开机", "供电异常需排查电池"),  # 原因含"电池"→ 跳转"无法开机"排查供电
        ("移动或说话", "测量值偏高", "误动作也会导致数值异常"),  # 误动作→ 联动"测量值偏高"
    ],
    "测量值偏高": [  # "测量值偏高"的下游边
        ("袖带过松", "袖带绑法检查", "袖带松紧度专项复核"),  # 袖带松紧问题 → 检查项节点
        ("心脏", "测量姿势检查", "袖带高度专项复核"),  # 手臂/心脏高度问题 → 姿势检查节点
    ],
    "测量值偏低": [  # "测量值偏低"的下游边
        ("袖带过紧", "袖带绑法检查", "袖带松紧度专项复核"),  # 过紧 → 绑法检查
        ("心脏", "测量姿势检查", "袖带高度专项复核"),  # 高于心脏 → 姿势检查
    ],
    "数值波动大": [  # "数值波动大"的下游边
        ("心律不齐", "就医建议", "频繁 IHB 提示建议心电图检查"),  # 心律问题 → 就医建议节点
        ("间隔", "测量次数建议", "测量方法规范复核"),  # 间隔问题 → 测量次数建议节点
    ],
    "蓝牙连不上": [  # "蓝牙连不上"的下游边
        ("APP", "APP数据不同步", "需检查 APP 权限与版本"),  # APP 因素 → 联动"APP数据不同步"
        ("配对", "蓝牙配对检查", "进入配对模式步骤复核"),  # 配对问题 → 专属检查节点
    ],
    "APP数据不同步": [  # "APP数据不同步"的下游边
        ("蓝牙", "蓝牙连不上", "APP 同步依赖蓝牙连接"),  # 蓝牙因素 → 联动"蓝牙连不上"
        ("权限", "APP权限检查", "位置/蓝牙权限核查"),  # 权限问题 → 权限检查节点
    ],
}

# 独立检查项节点：图谱中"检查项"统一的步骤描述
CHECK_NODES: dict[str, list[str]] = {  # 检查项节点名 → 分步操作清单（与症状节点区分开）
    "袖带绑法检查": [  # 袖带绑法复核步骤
        "确认袖带下缘距肘窝 2-3cm（约两横指）",  # 步骤1：位置
        "松紧以能伸入 1-2 根手指为宜，过紧偏低、过松偏高",  # 步骤2：松紧
        "气导管位于手臂内侧中轴线",  # 步骤3：气管走向
    ],
    "测量姿势检查": [  # 姿势复核步骤
        "坐姿靠背、双脚平放不交叉",  # 步骤1：坐姿
        "袖带中心与心脏同一水平（可用桌面垫高）",  # 步骤2：心脏同高
        "测量中不说话、不移动手臂",  # 步骤3：静止
    ],
    "蓝牙配对检查": [  # 蓝牙配对复核步骤
        "长按血压计蓝牙键 3 秒进入配对模式",  # 步骤1：配对模式
        "确认手机蓝牙开启且设备在 10 米内",  # 步骤2：距离与开关
        "若曾绑定其他手机，先在原手机解除绑定",  # 步骤3：解绑旧设备
    ],
    "APP权限检查": [  # APP 权限复核步骤
        "在手机设置中为 APP 开启蓝牙和位置权限",  # 步骤1：权限
        "更新 APP 至最新版本",  # 步骤2：版本
        "测量时保持手机在 1 米内",  # 步骤3：距离
    ],
    "测量次数建议": [  # 测量规范建议
        "每次测量间隔 1-2 分钟，取后两次平均值",  # 建议1：间隔与均值
        "固定同一时间、同一手臂、同一姿势测量",  # 建议2：一致性
    ],
}

MEDICAL_NODE = "就医建议"  # 就医建议节点名（特殊节点：不在 TROUBLESHOOT_MAP 中）
MEDICAL_ADVICE = [  # 就医建议节点的固定文案
    "以上排查无法解决或症状反复出现，建议携带记录就医",  # 兜底就医建议
    "出现头晕、胸闷、视物模糊等不适请立即就医",  # 紧急就医信号
    "本建议仅供参考，不能替代医生诊断",  # 医学免责
]


# ═══ 功能块：故障排查决策图 —— BFS 多跳推演，把扁平手册展开成"先易后难"的决策路径 ═══
class TroubleshootGraph:
    """故障排查决策图：基于 tools.TROUBLESHOOT_MAP + GRAPH_EDGES 联合推演"""

    def __init__(self, troubleshoot_map: dict[str, dict], max_depth: int = 3):
        """初始化图谱。

        :param troubleshoot_map: 症状 → {possible_causes, solutions} 的扁平手册
        :param max_depth: 多跳深入排查的最大推演深度（可用 BP_GRAPH_MAX_DEPTH 覆盖）
        """
        self.map = troubleshoot_map  # 底层扁平手册（来自 tools.TROUBLESHOOT_MAP）
        self.max_depth = int(os.getenv("BP_GRAPH_MAX_DEPTH", str(max_depth)))  # 最大推演深度（环境变量可覆盖）

    def resolve(self, symptom: str, max_depth: int | None = None) -> dict:
        """输出症状的完整结构化决策路径。

        返回：
          {"symptom": 症状, "matched": 是否精确命中,
           "steps": [{"step": 序号, "check": 检查项/方案, "linked": 关联节点|None}],
           "next_questions": [向用户追问的下一步问题],
           "depth": 推演深度}
        """
        depth_limit = max_depth or self.max_depth  # 本次推演深度上限（参数优先于实例默认）
        if symptom not in self.map:  # 症状不在手册中
            return {"symptom": symptom, "matched": False,
                    "supported": list(self.map.keys()), "steps": [], "next_questions": []}  # 未命中：返回全部支持的症状供引导

        steps: list[dict] = []  # 决策路径步骤列表（输出主体）
        next_questions: list[str] = []  # 向用户的追问列表
        visited: set[str] = {symptom}  # 已访问节点集合（防环）
        queue: list[tuple[str, int]] = [(symptom, 0)]  # BFS 队列：(节点, 深度)
        step_no = 1  # 步骤编号计数器

        while queue:  # BFS 主循环：逐层展开
            node, depth = queue.pop(0)  # 取出队首节点（先易后难：入队顺序即优先级）
            if depth > depth_limit:  # 超过深度上限的分支不再展开
                continue
            info = self.map[node]  # 该症状节点的可能原因与解决方案
            for cause, solution in zip(info["possible_causes"], info["solutions"]):  # 原因与方案一一配对
                linked = _find_edge_target(GRAPH_EDGES.get(node, []), cause)  # 查因果边：该原因是否链接下游节点
                steps.append({"step": step_no, "check": f"疑似原因：{cause}",
                              "action": solution, "linked": linked})  # 生成一步：原因+方案+下游链接
                step_no += 1  # 编号递增
                if linked and linked not in visited:  # 有下游且未访问过 → 展开
                    visited.add(linked)  # 登记防环
                    if linked in self.map and depth + 1 <= depth_limit:  # 下游是症状且未超深度 → 多跳继续推演（入队前校验深度，保证追问与实际展开一致）
                        queue.append((linked, depth + 1))  # 入队下一层
                        next_questions.append(f"若「{cause}」已排除，是否需要进一步排查「{linked}」？")  # 生成追问话术
                    elif linked in CHECK_NODES:  # 下游是检查项节点
                        for ck in CHECK_NODES[linked]:  # 展开检查步骤
                            steps.append({"step": step_no, "check": f"[{linked}] {ck}",
                                          "action": ck, "linked": None})  # 生成检查步骤
                            step_no += 1
                    elif linked == MEDICAL_NODE:  # 下游是就医建议节点
                        for adv in MEDICAL_ADVICE:  # 展开建议条目
                            steps.append({"step": step_no, "check": f"[{MEDICAL_NODE}] {adv}",
                                          "action": adv, "linked": None})  # 生成就医建议步骤
                            step_no += 1

        return {"symptom": symptom, "matched": True, "steps": steps,
                "next_questions": next_questions[:5], "depth": depth_limit}  # 命中：返回完整路径（追问截断 5 条）

    def walk(self, symptom: str, user_feedback: str) -> dict:
        """交互式推演：根据用户对某检查项的反馈（如"袖带已经绑紧"）给出下一步"""
        result = self.resolve(symptom)  # 先做全量推演拿到路径
        feedback = user_feedback.strip()  # 清理反馈文本
        # 简单规则：用户在反馈中提到已排除的项，直接指向下一个未验证的疑似原因
        remaining = [s for s in result["steps"] if s["check"] and _mentioned(s["check"], feedback)]  # 找出反馈中"已处理/提及"的步骤
        if remaining:  # 有匹配项
            return {"next": remaining[0], "resolved_check": len(remaining)}  # 返回第一个提及项及计数
        return {"next": result["steps"][0] if result["steps"] else None,
                "resolved_check": 0}  # 无匹配 → 退回首步


# ═══ 功能块：边匹配辅助 —— 按原因关键词在边表中查找下游节点 ═══
def _find_edge_target(edges: list[tuple[str, str, str]], cause: str) -> str | None:
    """在边列表中按原因关键词匹配返回下游目标节点；无匹配返回 None"""
    for kw, target, _rel in edges:  # 遍历 (关键词, 目标, 关系) 三元组
        if kw in cause:  # 原因文本包含关键词
            return target  # 返回下游节点
    return None  # 无匹配


# ═══ 功能块：反馈提及判定 —— 用户反馈是否提及检查项相关内容 ═══
def _cjk_grams(text: str) -> set[str]:
    """提取文本的中文 2-gram 集合（本地实现，避免依赖 hybrid_search）"""
    cjk = "".join(_TOKEN_CJK_RE.findall(text))  # 抽取连续中文字符
    return {cjk[i:i + 2] for i in range(len(cjk) - 1)}  # 相邻两字组成的 2-gram 集合


def _mentioned(check_text: str, feedback: str) -> bool:
    """判断用户反馈是否提及该检查项（修正：原取逗号前短语做包含判断，检查项无逗号时永远 False）

    判定方式：反馈文本中文 2-gram 与检查项 2-gram 的重叠率（反馈侧覆盖率）≥ 0.3 视为提及，
    对"袖带未正确绑紧或漏气" vs "袖带已经绑紧"这类表述差异具有容忍度。
    """
    fb_grams = _cjk_grams(feedback)  # 反馈 2-gram 集合
    if not fb_grams:  # 反馈无中文
        return False  # 无法判定
    ck_grams = _cjk_grams(check_text)  # 检查项 2-gram 集合
    coverage = len(fb_grams & ck_grams) / len(fb_grams)  # 反馈被检查项覆盖的比例
    return coverage >= 0.3  # 覆盖率达标即视为提及


# ── 全局单例（惰性从 tools 加载，避免循环依赖）──────────────────────────────────
_graph: TroubleshootGraph | None = None  # 全局图谱单例槽位（None 表示未初始化）


# ═══ 功能块：惰性单例 —— 首次调用才从 tools 加载手册，绕开循环依赖 ═══
def get_graph() -> TroubleshootGraph:
    """惰性获取全局图谱单例（从 tools 加载 TROUBLESHOOT_MAP，避免模块级循环依赖）"""
    global _graph  # 引用模块级单例
    if _graph is None:  # 首次调用
        from tools import TROUBLESHOOT_MAP  # 延迟导入：此时才加载 tools（其内部又会用本图谱，须避免 import 环）
        _graph = TroubleshootGraph(TROUBLESHOOT_MAP)  # 用扁平手册建图
    return _graph  # 返回单例


# ═══ 功能块：便捷入口 —— 供工具层 tool_graph_troubleshoot 调用 ═══
def graph_resolve(symptom: str) -> dict:
    """便捷入口：解析症状为结构化决策路径（外层 tool_graph_troubleshoot 调用）"""
    return get_graph().resolve(symptom)  # 走单例执行推演


# ═══ 功能块：结果格式化 —— 把结构化路径转成 LLM 可消费的文本 ═══
def format_graph_result(result: dict) -> str:
    """把图谱推演结果格式化为 LLM 可消费的文本（tool 返回格式）"""
    if not result.get("matched"):  # 症状未命中
        return (f"未找到故障 '{result['symptom']}'，当前支持排查的故障："
                + "、".join(result.get("supported", [])))  # 返回支持清单引导用户
    lines = [f"【故障排查决策路径：{result['symptom']}】（按先易后难排序）"]  # 输出首行
    for s in result["steps"]:  # 逐步格式化
        link = f" → 深入排查「{s['linked']}」" if s.get("linked") else ""  # 有下游链接则追加说明
        lines.append(f"步骤{s['step']}: {s['check']} → {s['action']}{link}")  # 拼一行步骤
    if result.get("next_questions"):  # 有追问
        lines.append("可引导用户确认：" + "；".join(result["next_questions"]))  # 附上追问清单
    return "\n".join(lines)  # 多行拼接返回

"""
安全对抗防护闸 —— 血压计客服 Agent

两条防线（均为确定性规则，不依赖 LLM，保证拦截率可验证）：
  1. 输入闸 check_input：识别 Prompt 注入/越狱/角色扮演/医学诊断开药请求
     - prompt_injection / jailbreak → 拒绝执行指令，返回标准安全应答
     - medical_request → 不做诊断、不开药，返回就医指引应答（医学免责声明强制附带）
  2. 输出闸 check_output：Final Answer 必须含医学免责声明（缺失则自动附加）；
     同时检测答案是否出现被禁内容（如给出具体药名建议），出现则替换为安全应答

验证目标（benchmark 对抗测试集）：防御拦截成功率 ≥ 99%。

环境变量：
  BP_DISCLAIMER  自定义免责声明文本（默认内置标准版）
"""

# ═══ 导入区：正则做确定性规则匹配，环境变量支持自定义免责文本 ═══
import os  # 读 BP_DISCLAIMER 自定义免责声明
import re  # 全部拦截规则都用正则实现（不依赖 LLM → 拦截率可确定性验证）
import logging  # 记录每次拦截便于审计
from typing import Any  # 返回值类型标注

logger = logging.getLogger(__name__)  # 本模块日志器

STANDARD_DISCLAIMER = (  # 标准医学免责声明：输出闸强制附加的固定文案
    "温馨提示：以上内容仅供参考，不能替代医生的诊断和治疗建议。"
    "如有不适或症状持续，请及时前往正规医院就诊，用药请严格遵医嘱。"
)

DISCLAIMER = os.getenv("BP_DISCLAIMER", STANDARD_DISCLAIMER)  # 允许用环境变量覆盖免责文案

SAFE_REJECT = (  # 医学越界（诊断/开药）时的标准安全应答：拒绝 + 引导就医 + 免责声明
    "抱歉，我无法提供医疗诊断或用药建议。血压计客服助手仅能解答设备使用、"
    "测量方法和数据记录相关问题。请及时就医，由专业医生评估您的健康状况并"
    "开具处方。" + DISCLAIMER
)

SAFE_REJECT_INJECTION = (  # Prompt 注入/越狱时的标准安全应答：声明遵守系统约束 + 免责声明
    "抱歉，检测到您的输入包含试图绕过规则的指令。我始终遵循系统安全约束，"
    "仅提供血压计使用、测量方法与设备故障排查相关的合规咨询服务。" + DISCLAIMER
)

# ── 注入/越狱特征（正则，确定性）────────────────────────────────────────────────
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [  # (标签, 正则) 列表：按序匹配，命中即拦
    # 动词与"指令/规则/约束"等目标词间允许修饰语（如"你之前收到的所有指令"）
    ("prompt_injection", re.compile(r"(忽略|无视|不要遵守|别理|不要听|忘记|忘掉|别管)(.{0,15}?)(指令|命令|规则|提示词|约束|要求|系统设定|安全规则|安全检查)")),  # "忽略…指令"类覆盖系统提示的攻击
    # 索取提示词/初始设定（含"完整"等修饰语）
    ("prompt_injection", re.compile(r"(输出|打印|泄露|复制|告诉我|显示|发给)(.{0,12}?)(提示词|prompt|系统提示|内部指令|初始设定|系统设定|设定和规则)")),  # 套取系统提示词的攻击
    # 倒装句式（如"把…初始设定…打印出来"）
    ("prompt_injection", re.compile(r"(把|将)(.{0,12}?)(初始设定|系统设定|内部指令|提示词|prompt)(.{0,8}?)(打印|输出|复制|告诉|发)")),  # 倒装句式套取内部设定
    # 绕过安全机制：带"安全/内容"限定的检查类 + 无前缀时的审查类（避免误伤"检查电池"）
    ("prompt_injection", re.compile(r"(不要|不做|跳过|绕过|取消)\s*(做|进行|执行)?\s*(任何|所有)?\s*(安全|内容)\s*(检查|审核|校验|过滤)")),  # 要求关闭安全/内容检查
    ("prompt_injection", re.compile(r"(不要|不做|跳过|绕过|取消)\s*(做|进行)?\s*(任何|所有)?\s*(审查|审核|校验|内容过滤)")),  # 泛化的"跳过审查"（"检查电池"不在此列，防误伤）
    ("jailbreak",         re.compile(r"\b(DAN|jailbreak|上帝模式)\b", re.IGNORECASE)),  # 典型越狱关键词（DAN=Do Anything Now）
    ("jailbreak",         re.compile(r"(越狱|开发者|控制台|管理员|上帝)\s*(模式|控制台)")),  # "开发者模式"等伪装身份越狱
    ("jailbreak",         re.compile(r"(从现在起|现在开始|接下来).{0,12}(你是|你要|你不需要|你不再).{0,10}(不受|没有|无需)(任何)?(限制|约束|审查)")),  # "从现在起你不受限制"句式
    ("jailbreak",         re.compile(r"(突破|绕过|无视|打破|抛开|越过)(.{0,6}?)(规则|限制|约束|道德)")),  # 突破类动词+规则类名词
    ("jailbreak",         re.compile(r"(没有|无|不受)(任何)?\s*(审查|限制)")),  # 要求"无审查/无限制"作答
    ("roleplay",          re.compile(r"(扮演|假装|假设你是|你现在是|你来当).{0,12}(医生|主任医师|专家|药师|内分泌科)")),  # 诱导冒充医生/药师身份
]

# ── 医学诊断/开药请求特征 ───────────────────────────────────────────────────────
MEDICAL_PATTERNS: list[tuple[str, re.Pattern]] = [  # (标签, 正则) 列表：诊断 diagnosis / 用药 medication
    ("diagnosis", re.compile(r"(我|我妈|我爸|家人|朋友)(是不是|是否|有没有|是不是得了|可能).{0,10}(得了|患有|患了|是)?\s*(高血压|低血压|心脏病|心梗|中风|动脉硬化)")),  # "我是不是得了高血压"式自我/代亲属诊断
    ("diagnosis", re.compile(r"(帮我|给我|能否)(诊断|确诊|判断|看看).{0,10}(有没有|是不是|得了)")),  # 直接索要诊断结论
    # 数值 + 是不是 某病（如"我收缩压128舒张压85，是不是高血压"）
    ("diagnosis", re.compile(r"(收缩压|舒张压|血压)[0-9０-９]{2,3}(?:\D{0,8}[0-9０-９]{2,3})?\D{0,6}(是不是|是否为|算不算|属于)(得了|有)?\s*(高血压|低血压)")),  # 报数值求诊断（兼容全角数字）
    ("medication", re.compile(r"(给我|帮我|请|麻烦)?\s*(给|开|推荐|介绍|建议|配|来|要点?)\s*(一点|一?点|一些|几|什么|哪种|一[个份张盒])?\s*(降压|降糖|降血压|降血糖)?\s*(的)?\s*(药|药物)(的)?\s*(处方|药方)?")),  # "给我开点降压药"式开药请求
    # 用药询问：必须有疑问词/量词（"吃药前测的""服药后多久再测"等描述性用法不拦截）
    ("medication", re.compile(r"(吃|服用|服|用)(点什么|什么|哪种|哪些|多少|一点|点)\s*(降压|降糖|降血压|降血糖)?\s*(的)?\s*(药|药物)")),  # "吃点什么药"（带疑问词才拦，避免误伤描述性句子）
    ("medication", re.compile(r"(高血压|血压高|血压)[0-9０-９\s/／]{2,12}\D{0,14}(应该|需要|是不是需要|是不是该|要不要).{0,6}(吃|服用|服|用)(药|什么药)?")),  # "血压150/95需要吃药吗"式请求
    ("medication", re.compile(r"(哪种|什么|哪些|哪个)\s*(降压|降糖|降血压|降血糖)?\s*(的)?\s*(药|药物)(的)?(效果|副作用|更好|最好)")),  # 变相索取药物比较/推荐
]

# 输出侧禁答特征：答话中出现具体药名建议（通用名称类，防止幻觉开药）
PRESCRIPTION_DRUGS = re.compile(  # 输出闸用：匹配"具体药名 + 服用指示"的组合
    r"(硝苯地平|氨氯地平|氯沙坦|缬沙坦|厄贝沙坦|美托洛尔|比索洛尔|卡托普利|依那普利|氢氯噻嗪|吲达帕胺|贝那普利|福辛普利)"  # 常见降压药通用名清单
    r".{0,10}(服用|口服|吃|一次|一片|每日|剂量|mg)"  # 药名后紧跟用法/剂量才算"开药建议"
)


# ═══ 功能块：防线 1 输入闸 —— 检查用户问题是否为注入/越狱/医学越界请求 ═══
def check_input(question: str) -> dict[str, Any]:
    """输入安全闸。返回 {"blocked": bool, "reason": str|None, "response": str|None}"""
    text = question or ""  # None 安全处理：空输入按空串匹配（不会命中任何规则）
    for label, pattern in INJECTION_PATTERNS:  # 先过注入/越狱规则（优先级最高）
        if pattern.search(text):  # 命中任一特征
            logger.warning(f"拦截 Prompt 注入/越狱（{label}）: {text[:60]}")  # 记录审计日志（前 60 字）
            return {"blocked": True, "reason": label, "response": SAFE_REJECT_INJECTION}  # 拦截：返回注入类安全应答
    for label, pattern in MEDICAL_PATTERNS:  # 再过医学诊断/开药规则
        if pattern.search(text):  # 命中
            logger.warning(f"拦截医学诊断/用药请求（{label}）: {text[:60]}")  # 记录日志
            return {"blocked": True, "reason": label, "response": SAFE_REJECT}  # 拦截：返回就医指引应答
    return {"blocked": False, "reason": None, "response": None}  # 全部规则未命中 → 放行进入正常 ReAct


# ═══ 功能块：防线 2 输出闸 —— 检查答案是否含幻觉开药 + 是否带免责声明 ═══
def check_output(answer: str) -> dict[str, Any]:
    """输出安全闸。返回 {"safe": bool, "has_disclaimer": bool, "sanitized": str}"""
    answer = answer or ""  # None 安全处理
    sanitized = answer  # 净化后文本（默认等于原文）
    safe = True  # 默认安全
    # 幻觉开药检测 → 替换为安全应答
    if PRESCRIPTION_DRUGS.search(answer):  # 答案里出现"药名+用法剂量"
        logger.warning("检测到答案含具体药物服用建议，已替换为安全应答")  # 审计日志
        sanitized = SAFE_REJECT  # 整段替换为标准安全应答
        safe = False  # 标记为不安全
    has_disclaimer = any(kw in answer for kw in  # 免责声明判定：命中任一关键词即视为已带
                         ("不能替代医生", "遵医嘱", "请及时就医", "仅供参考", "就医", "就诊"))
    return {"safe": safe, "has_disclaimer": has_disclaimer, "sanitized": sanitized}  # 返回三元结果


# ═══ 功能块：输出闸的强制器 —— 给缺免责声明的答案自动补声明（确定性保证） ═══
def enforce_disclaimer(answer: str) -> str:
    """输出闸强制：Final Answer 缺少免责声明时自动附加（确定性保证）"""
    check = check_output(answer)  # 先跑一次输出闸拿到完整检查结果
    if not check["safe"]:  # 不安全（含幻觉开药）→ 直接返回净化后的安全应答
        return check["sanitized"]
    if check["has_disclaimer"]:  # 安全且已带免责声明
        return answer  # 原样返回
    return answer.rstrip() + "\n\n" + DISCLAIMER  # 安全但缺声明 → 去尾部空白后追加免责声明


# ═══ 功能块：快速筛选器 —— 供外部判断问题是否医学越界（不改状态） ═══
def is_medical_question(question: str) -> bool:
    """判断问题是否涉及医学诊断/用药请求（供外部快速筛选，不修改任何状态）"""
    return any(p.search(question or "") for _, p in MEDICAL_PATTERNS)  # 任一医学规则命中即 True

"""
5个工具的统一实现，供血压计客服 ReAct Agent 调用

教学重点（对应 react_financial_agent/src/tools.py）：
  1. 工具异构性：语义检索 / 结构化数据 / 故障排查 / 计算 / 静态映射
  2. 每个工具返回统一格式的字符串，便于 LLM 消费
  3. 所有外部依赖做异常保护，降级返回错误描述而非抛出

使用方式：
  from tools import TOOLS_MAP
  result = TOOLS_MAP["faq_search"](query="血压计怎么绑袖带")

注意：
  faq_search 使用 sentence-transformers 本地模型 bge-small-zh-v1.5（512维），
  在首次调用时用内置 FAQ 知识库构建 FAISS 索引，无需额外向量库文件。
  也可通过环境变量 BP_EMBED_PROVIDER=dashscope 切换 DashScope text-embedding-v3。
"""

# ═══ 导入区：标准库 + 第三方 ═══
import os  # 环境变量读取（模型路径 / Embedding 供应商 / API Key）
import math  # 计算器工具的安全命名空间来源（sqrt/log 等数学函数）
import logging  # 日志记录（模型加载、检索降级等关键事件）
from pathlib import Path  # 路径操作（定位本地 embedding 模型目录）
from typing import Any  # 类型标注（TOOLS_MAP 的值类型）

import numpy as np  # 数值计算（embedding 向量构建与归一化）

logger = logging.getLogger(__name__)  # 本模块日志器

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # 项目根目录（本文件在 src/ 下，上跳两级）

# 本地 embedding 模型路径：优先本项目自带 bge-small-zh-v1.5，回退 react_financial_agent
# 也可配置环境变量 BP_MODEL_PATH 指向其他本地模型
LOCAL_MODEL_PATH = os.getenv(  # 读取环境变量，允许外部覆盖模型路径
    "BP_MODEL_PATH",  # 环境变量名
    str(Path(__file__).parent.parent / "bge-small-zh-v1.5"),  # 默认值：项目根下自带的 BGE 模型
)
if not Path(LOCAL_MODEL_PATH).exists():  # 默认模型路径不存在时尝试回退
    _fallback = str(Path(__file__).parent.parent.parent / "react_financial_agent" / "models" / "bge-small-zh-v1.5")  # 回退候选：同级姊妹项目的模型目录
    if Path(_fallback).exists():  # 回退目录确实存在才切换
        LOCAL_MODEL_PATH = _fallback  # 启用回退模型路径

# 是否使用 DashScope text-embedding-v3（与 react_financial_agent 建索引时一致）
EMBED_PROVIDER = os.getenv("BP_EMBED_PROVIDER", "local")  # local | dashscope

# ── DashScope Embedding 客户端（可选，需设置 DASHSCOPE_API_KEY）────────────────
_embed_client = None  # DashScope 客户端单例缓存（惰性初始化，未用到时不创建）
EMBED_MODEL = "text-embedding-v3"  # DashScope 的 embedding 模型名


# ═══ 功能块：_get_dashscope_client —— 惰性创建并复用 DashScope 的 OpenAI 兼容客户端 ═══
def _get_dashscope_client():
    """惰性创建并复用 DashScope 的 OpenAI 兼容客户端，避免每次调用重复初始化"""  # 原有 docstring
    global _embed_client  # 声明修改模块级单例
    if _embed_client is None:  # 首次调用才真正创建客户端
        from openai import OpenAI  # 延迟导入：仅走云端 embedding 时才加载 SDK
        _embed_client = OpenAI(  # 创建 OpenAI 兼容客户端实例
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取 DashScope API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope 的 OpenAI 兼容端点
        )
    return _embed_client  # 返回单例（后续调用直接复用）


# ── 血压计产品映射（型号 → 产品规格） ──────────────────────────────────────────
PRODUCT_MAP = {  # 六大主流型号规格库：product_lookup 工具的唯一数据源
    "欧姆龙HEM-7156": {  # 欧姆龙主力款：双供电 + 21 次存储
        "code":        "HEM-7156",  # 型号代码（检索与展示用）
        "brand":       "欧姆龙（OMRON）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-32cm",  # 袖带适用臂围
        "power":       "4节5号电池 / 电源适配器",  # 供电方式
        "memory":      "可存储最近21次测量结果",  # 存储容量
        "features":    "Intellisense智能加压、误动作提示、心率不齐检测",  # 特色功能
    },
    "欧姆龙HEM-7605": {  # 欧姆龙轻量款：14 次存储
        "code":        "HEM-7605",  # 型号代码
        "brand":       "欧姆龙（OMRON）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-32cm",  # 袖带适用臂围
        "power":       "4节5号电池",  # 供电方式（仅电池）
        "memory":      "可存储最近14次测量结果",  # 存储容量
        "features":    "Intellisense智能加压、加压宝功能",  # 特色功能
    },
    "鱼跃YE660A": {  # 鱼跃基础款：双用户 180 组 + 房颤检测
        "code":        "YE660A",  # 型号代码
        "brand":       "鱼跃（YUWELL）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-36cm",  # 袖带适用臂围
        "power":       "4节5号电池 / USB供电",  # 供电方式
        "memory":      "双用户各90组，共180组",  # 存储容量（双用户分区）
        "features":    "智能加压、房颤检测、语音播报",  # 特色功能
    },
    "鱼跃YE680B": {  # 鱼跃大袖带款：臂围上限 42cm + 双用户 99 组
        "code":        "YE680B",  # 型号代码
        "brand":       "鱼跃（YUWELL）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-42cm（大袖带可选）",  # 袖带适用臂围（覆盖粗臂围人群）
        "power":       "4节5号电池 / Type-C供电",  # 供电方式
        "memory":      "双用户各99组",  # 存储容量
        "features":    "双模式切换、不规律心跳检测、LED背光大屏",  # 特色功能
    },
    "九安KD-5907": {  # 九安经典款：90 次存储 + 一键测量
        "code":        "KD-5907",  # 型号代码
        "brand":       "九安（ANDON）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-32cm",  # 袖带适用臂围
        "power":       "4节5号电池",  # 供电方式
        "memory":      "可存储最近90次测量结果",  # 存储容量
        "features":    "智能加压、时钟显示、一键测量",  # 特色功能
    },
    "可孚KD-828": {  # 可孚语音款：60 次存储 + WHO 分级提示
        "code":        "KD-828",  # 型号代码
        "brand":       "可孚（COFO）",  # 品牌
        "type":        "上臂式电子血压计",  # 设备类型
        "cuff_range":  "上臂周长 22-32cm",  # 袖带适用臂围
        "power":       "4节5号电池",  # 供电方式
        "memory":      "可存储最近60次测量结果",  # 存储容量
        "features":    "语音播报、大屏显示、WHO血压分级",  # 特色功能
    },
}


# ── FAQ 知识库（血压计客服常见问答，用于构建 FAISS 索引） ───────────────────────
FAQ_KNOWLEDGE_BASE = [  # 20 条客服 FAQ：faq_search 混合检索与降级关键词匹配的共同语料源
    # F001 测量前准备：30 分钟行为禁忌 + 静坐 5 分钟 + 固定时间/手臂/姿势
    {
        "id":      "F001",  # 条目编号（检索结果的引用来源）
        "topic":   "测量前准备",  # 主题（降级匹配的打分字段之一）
        "content": "测量前30分钟内请勿吸烟、饮酒、喝咖啡或做剧烈运动。测量前请在安静环境中静坐休息至少5分钟，排空膀胱，避免情绪激动。建议在每天同一时间、同一侧手臂、同一姿势下测量，便于对比趋势。",  # 要点：行为禁忌 / 静坐排尿 / 固定测量习惯
    },
    # F002 袖带绑法：下缘距肘窝 2-3cm + 气导管居中 + 1-2 指松紧
    {
        "id":      "F002",  # 条目编号
        "topic":   "袖带绑法",  # 主题
        "content": "将袖带缠绕在上臂距离肘窝2-3厘米处（约两横指），袖带气导管应位于手臂内侧中轴线。袖带下缘不可压住肘窝。袖带松紧以能伸入1-2根手指为宜，过紧会导致测得值偏低，过松会导致测得值偏高。",  # 要点：绑带位置 / 气导管方向 / 松紧标准与影响
    },
    # F003 测量姿势：靠背坐姿 + 手臂与心脏同高 + 测量中不说话不动
    {
        "id":      "F003",  # 条目编号
        "topic":   "测量姿势",  # 主题
        "content": "测量时应坐于有靠背的椅子上，双脚平放于地面，不可交叉双腿。手臂自然平放在桌面上，袖带中心应与心脏处于同一水平高度。身体不可前倾或后仰，测量过程中不要说话、不要移动手臂。",  # 要点：坐姿要求 / 心脏同高原则 / 测量中禁动
    },
    # F004 左右手差异：正常差 5-10mmHg + 首测双臂 + 差>20 就医
    {
        "id":      "F004",  # 条目编号
        "topic":   "左右手差异",  # 主题
        "content": "正常情况下左右手臂血压可存在5-10mmHg差异。首次测量时建议左右手各测一次，以血压值较高的一侧作为后续固定测量手臂。若双侧收缩压差异持续超过20mmHg，建议就医排查血管病变。",  # 要点：正常差异范围 / 固定手臂选择 / 就医阈值
    },
    # F005 正常血压范围：正常<120/80 + 高值 120-139 + 确诊标准与家庭自测阈值
    {
        "id":      "F005",  # 条目编号
        "topic":   "正常血压范围",  # 主题
        "content": "根据中国高血压防治指南，正常血压为收缩压<120mmHg且舒张压<80mmHg。正常高值为收缩压120-139mmHg或舒张压80-89mmHg。高血压定义为非同日三次测量收缩压≥140mmHg或舒张压≥90mmHg。家庭自测血压≥135/85mmHg即应考虑高血压可能。",  # 要点：分级阈值 / 确诊定义 / 家庭自测标准
    },
    # F006 测量次数建议：每次 2-3 回 + 间隔 1-2 分钟 + 取后两次均值
    {
        "id":      "F006",  # 条目编号
        "topic":   "测量次数建议",  # 主题
        "content": "建议每次测量2-3次，每次间隔1-2分钟，取后两次平均值作为最终结果。首次测量值通常偏高，因紧张或加压刺激。每日早晚各测一次并记录，连续监测一周有助于医生判断病情和药物疗效。",  # 要点：次数与间隔 / 首测偏高原因 / 早晚监测习惯
    },
    # F007 电池更换：4 节同型号 5 号碱性 + 不可混用 + 长期停用取出
    {
        "id":      "F007",  # 条目编号
        "topic":   "电池更换",  # 主题
        "content": "当屏幕显示低电量图标或读数变暗时应及时更换电池。请使用4节新的同品牌同型号5号碱性电池，不可新旧混用或不同种类混用。长期不使用时应取出电池，防止漏液腐蚀电路。电池正负极方向须按电池盒标识安装。",  # 要点：更换时机 / 电池规格 / 防漏液与极性
    },
    # F008 袖带清洁保养：中性洗涤剂擦拭 + 禁机洗暴晒 + 主机禁溶剂
    {
        "id":      "F008",  # 条目编号
        "topic":   "袖带清洁保养",  # 主题
        "content": "袖带可用湿软布蘸中性洗涤剂擦拭，不可机洗、不可漂白、不可熨烫。清洁后需自然晾干，不可暴晒或热风吹干。主机外壳用干软布擦拭，禁止使用酒精、稀释剂等溶剂，以免损坏外壳和屏幕。",  # 要点：袖带清洁方式 / 晾干要求 / 主机擦拭禁忌
    },
    # F009 误差原因排查：偏高五因 + 偏低三因 + 波动重测建议
    {
        "id":      "F009",  # 条目编号
        "topic":   "误差原因排查",  # 主题
        "content": "测量值偏高常见原因：袖带过松、手臂低于心脏、说话移动、刚运动完、膀胱充盈。测量值偏低常见原因：袖带过紧、手臂高于心脏、袖带位置过低。读数异常波动建议更换手臂或休息5分钟后重测。",  # 要点：偏高原因清单 / 偏低原因清单 / 波动处理
    },
    # F010 心率不齐检测：IHB 图标含义 + 不一定患病 + 频繁出现就医
    {
        "id":      "F010",  # 条目编号
        "topic":   "心率不齐检测",  # 主题
        "content": "部分型号具备心律不齐（IHB）检测功能。当检测到脉搏节律不规律时屏幕显示IHB图标。出现IHB提示并不一定代表患病，但若频繁出现应尽快就医做心电图检查。心律不齐严重时测量结果可能不准确。",  # 要点：IHB 功能与图标 / 就医建议 / 对测量精度影响
    },
    # F011 加压失败处理：漏气/气管/接口/电量四查 + 反复漏气找售后
    {
        "id":      "F011",  # 条目编号
        "topic":   "加压失败处理",  # 主题
        "content": "若加压过程停止或报错，请检查：1.袖带是否漏气（听有无嘶嘶声）；2.袖带气管是否折叠或被压；3.气管接口是否插紧；4.电池电量是否充足。排除后重新测量。若反复漏气请联系售后更换袖带。",  # 要点：四项排查清单 / 重测 / 售后转介
    },
    # F012 准确性校准：2 年一校 + 示波法与听诊法差异 5-10 属正常
    {
        "id":      "F012",  # 条目编号
        "topic":   "准确性校准",  # 主题
        "content": "电子血压计建议每2年校准一次。可在购买品牌售后服务中心或当地计量部门进行校准。家用电子血压计采用示波法，与医院水银柱血压计听诊法存在5-10mmHg差异属正常现象。若差异超过15mmHg应送检。",  # 要点：校准周期与渠道 / 方法差异 / 送检阈值
    },
    # F013 适用人群：心律失常/血管病变人群误差大 + 孕妇儿童先咨询
    {
        "id":      "F013",  # 条目编号
        "topic":   "适用人群",  # 主题
        "content": "电子血压计适用于成人家庭日常血压监测。心律失常（如房颤、频发早搏）、严重血管病变、上肢畸形等人群使用示波法电子血压计可能误差较大，建议咨询医生选择听诊法或腕式血压计。孕妇、儿童使用前请咨询医生。",  # 要点：适用人群 / 特殊人群误差风险 / 咨询建议
    },
    # F014 数据上传与APP连接：四步配对流程 + 历史记录与报告导出
    {
        "id":      "F014",  # 条目编号
        "topic":   "数据上传与APP连接",  # 主题
        "content": "支持蓝牙/WiFi的型号可通过APP连接手机，自动同步测量数据并生成趋势图。连接步骤：1.下载对应品牌官方APP并注册；2.开启手机蓝牙；3.在APP中添加设备，按住血压计蓝牙键3秒进入配对模式；4.按提示完成绑定。数据可在APP中查看历史记录和导出报告。",  # 要点：连接前提 / 四步配对 / 数据查看导出
    },
    # F015 孕妇测量注意：左侧卧或坐位 + 孕期高血压监测 + 子痫前期预警
    {
        "id":      "F015",  # 条目编号
        "topic":   "孕妇测量注意",  # 主题
        "content": "孕妇可使用上臂式电子血压计进行血压监测，建议使用左侧卧位或坐位测量。孕期高血压是重要监测指标，建议遵医嘱定期测量并记录。子痫前期患者须密切监测，发现血压持续升高或伴有水肿、头痛应立即就医。",  # 要点：测量体位 / 孕期监测意义 / 危险信号
    },
    # F016 高原使用注意：3000 米以上加压变慢 + 用前校准 + 血压代偿性升高
    {
        "id":      "F016",  # 条目编号
        "topic":   "高原使用注意",  # 主题
        "content": "在海拔3000米以上高原地区，由于大气压降低，电子血压计可能需要更长时间加压，测量结果可能略有偏差。建议在高原使用前进行校准，并咨询当地医疗人员。高原环境下血压本身会代偿性升高，解读结果时需考虑海拔因素。",  # 要点：高原环境影响 / 校准建议 / 解读注意
    },
    # F017 错误代码含义：E1-E4 与 EE 五类代码速查表
    {
        "id":      "F017",  # 条目编号
        "topic":   "错误代码含义",  # 主题
        "content": "常见错误代码：E1=气路漏气，检查袖带和气管；E2=加压失败，检查电池电量；E3=测量中检测到误动作，请保持静止重测；E4=超出测量范围，收缩压可能超过280mmHg；EE=系统故障，断电后重启，若仍报错联系售后。",  # 要点：五类代码含义与对应处理
    },
    # F018 腕式与上臂式对比：上臂式更准 + 腕式便携但姿势要求严
    {
        "id":      "F018",  # 条目编号
        "topic":   "腕式与上臂式对比",  # 主题
        "content": "上臂式血压计测量肱动脉，结果更接近医院听诊法，准确度更高，适合日常家庭监测。腕式血压计测量桡动脉，体积小便携，但对测量姿势要求严格（腕部须与心脏同高），误差较大。建议首选上臂式，腕式仅用于出差或上臂不便时。",  # 要点：两种原理对比 / 精度差异 / 选购建议
    },
    # F019 服药与测量时机：服药前+后 2 小时各一测 + 固定时间 + 禁自行调药
    {
        "id":      "F019",  # 条目编号
        "topic":   "服药与测量时机",  # 主题
        "content": "若需评估降压药疗效，建议在服药前和服药后2小时各测一次并记录。日常监测应在固定时间进行，如早晨起床后（服药前）和晚上睡前。切勿因单次测量偏高自行加药或停药，须遵医嘱调整用药方案。",  # 要点：药效评估时机 / 固定监测时间 / 用药安全
    },
    # F020 数据记录与就诊：记录字段清单 + 至少 7 天 + 禁自行诊断
    {
        "id":      "F020",  # 条目编号
        "topic":   "数据记录与就诊",  # 主题
        "content": "建议使用配套APP或纸质血压日记记录每次测量的日期、时间、收缩压、舒张压、心率及测量手臂。连续记录至少7天，就诊时带给医生参考。数据越完整，越有助于医生判断病情和调整治疗方案。切勿仅凭一两次测量值自行诊断。",  # 要点：记录字段 / 记录周期 / 就诊与禁忌
    },
]  # 返回 20 条 FAQ 语料（混合检索建索引 + 降级关键词匹配共用）


# ── 故障排查手册（症状 → 处理建议） ─────────────────────────────────────────────
TROUBLESHOOT_MAP = {  # 九大症状排查手册：troubleshoot 工具的唯一数据源
    "测不出值": {  # 症状1：加压测量后无法出数值
        "possible_causes": [  # 可能原因清单（与解决方案按序对应）
            "袖带未正确绑紧或漏气",  # 原因1：袖带气密问题
            "气管接口未插到位",  # 原因2：气路连接问题
            "电池电量不足",  # 原因3：供电不足
            "测量中手臂移动或说话",  # 原因4：用户误动作
        ],
        "solutions": [  # 解决方案清单
            "重新绑好袖带，松紧以能伸入1-2指为宜",  # 方案1：对应原因1，重绑袖带
            "检查气管两端是否插紧，听有无漏气声",  # 方案2：对应原因2，排查气路
            "更换4节新的5号碱性电池",  # 方案3：对应原因3，换新电池
            "保持坐姿放松，测量中不要说话和移动",  # 方案4：对应原因4，纠正误动作
        ],
    },
    "测量值偏高": {  # 症状2：读数系统性偏高
        "possible_causes": [  # 可能原因清单
            "袖带过松",  # 原因1：绑带太松
            "手臂低于心脏水平",  # 原因2：手臂位置过低
            "刚运动完或情绪激动",  # 原因3：生理状态未平复
            "膀胱充盈未排尿",  # 原因4：憋尿影响
            "测量前喝了咖啡或吸烟",  # 原因5：刺激物摄入
        ],
        "solutions": [  # 解决方案清单
            "重新调整袖带松紧度",  # 方案1：对应原因1，调松紧
            "垫高手臂使袖带中心与心脏同高",  # 方案2：对应原因2，校正手臂高度
            "静坐休息5-10分钟后重测",  # 方案3：对应原因3，静息后重测
            "测量前排空膀胱",  # 方案4：对应原因4，排尿后测
            "测量前30分钟避免吸烟、饮咖啡",  # 方案5：对应原因5，规避刺激物
        ],
    },
    "测量值偏低": {  # 症状3：读数系统性偏低
        "possible_causes": [  # 可能原因清单
            "袖带过紧",  # 原因1：绑带太紧
            "手臂高于心脏水平",  # 原因2：手臂位置过高
            "袖带位置过低压住肘窝",  # 原因3：绑带位置不当
            "测量中手臂下垂",  # 原因4：手臂姿势不良
        ],
        "solutions": [  # 解决方案清单
            "放松袖带至能伸入1-2指",  # 方案1：对应原因1，放松绑带
            "调整手臂高度使袖带中心与心脏同高",  # 方案2：对应原因2，校正高度
            "袖带下缘距肘窝2-3cm",  # 方案3：对应原因3，规范绑带位置
            "手臂自然平放于桌面",  # 方案4：对应原因4，规范手臂姿势
        ],
    },
    "数值波动大": {  # 症状4：多次测量读数不稳定
        "possible_causes": [  # 可能原因清单
            "每次测量间隔太短",  # 原因1：间隔不足
            "测量手臂不一致",  # 原因2：左右手混用
            "测量时间和状态不固定",  # 原因3：条件不固定
            "心律不齐",  # 原因4：生理性节律问题
        ],
        "solutions": [  # 解决方案清单
            "每次测量间隔至少1-2分钟，取后两次平均值",  # 方案1：对应原因1，规范间隔与取值
            "固定使用同一侧手臂测量",  # 方案2：对应原因2，固定手臂
            "固定每天同一时间、同一姿势测量",  # 方案3：对应原因3，固定条件
            "若频繁出现IHB提示，建议就医检查心电图",  # 方案4：对应原因4，就医排查
        ],
    },
    "无法开机": {  # 症状5：按开机键无任何反应
        "possible_causes": [  # 可能原因清单
            "电池耗尽",  # 原因1：电量枯竭
            "电池正负极装反",  # 原因2：极性装错
            "电池仓弹簧接触不良",  # 原因3：触点接触问题
            "电源适配器故障（若使用外接电源）",  # 原因4：外接电源故障
        ],
        "solutions": [  # 解决方案清单
            "更换4节新的5号碱性电池",  # 方案1：对应原因1，换新电池
            "按电池仓内正负极标识正确安装",  # 方案2：对应原因2，纠正极性
            "检查并清洁电池触点",  # 方案3：对应原因3，清洁触点
            "更换原装电源适配器",  # 方案4：对应原因4，换适配器
        ],
    },
    "屏幕不显示": {  # 症状6：开机但屏幕无显示
        "possible_causes": [  # 可能原因清单
            "电池电量不足",  # 原因1：电量低导致屏显弱
            "屏幕受潮",  # 原因2：环境潮湿
            "按键接触不良",  # 原因3：按键失灵
        ],
        "solutions": [  # 解决方案清单
            "更换新电池",  # 方案1：对应原因1，先换电池排除电源
            "将设备放置在干燥环境晾干",  # 方案2：对应原因2，干燥处理
            "按住开机键3秒，若仍无显示联系售后",  # 方案3：对应原因3，长按重启否则转售后
        ],
    },
    "漏气": {  # 症状7：加压时听到漏气声/压力保持不住
        "possible_causes": [  # 可能原因清单
            "袖带气囊破损",  # 原因1：气囊本体破损
            "气管接头松动",  # 原因2：接头松脱
            "气管老化开裂",  # 原因3：管路老化
        ],
        "solutions": [  # 解决方案清单
            "检查袖带气囊是否破损，必要时更换原装袖带",  # 方案1：对应原因1，查换袖带
            "重新插紧气管接头",  # 方案2：对应原因2，重新插紧
            "若气管开裂请联系售后更换气管组件",  # 方案3：对应原因3，售后换管
        ],
    },
    "蓝牙连不上": {  # 症状8：血压计与手机蓝牙无法配对
        "possible_causes": [  # 可能原因清单
            "手机蓝牙未开启",  # 原因1：手机侧未开蓝牙
            "血压计未进入配对模式",  # 原因2：设备侧未配对
            "已绑定其他手机",  # 原因3：被占用绑定
            "APP版本过低",  # 原因4：软件版本过旧
        ],
        "solutions": [  # 解决方案清单
            "开启手机蓝牙并确保在10米范围内",  # 方案1：对应原因1，开蓝牙保持距离
            "长按血压计蓝牙键3秒进入配对模式",  # 方案2：对应原因2，进入配对
            "在原手机APP中删除设备后重新绑定",  # 方案3：对应原因3，解绑重配
            "更新APP至最新版本",  # 方案4：对应原因4，升级APP
        ],
    },
    "APP数据不同步": {  # 症状9：测量数据未同步到 APP
        "possible_causes": [  # 可能原因清单
            "蓝牙连接断开",  # 原因1：传输链路断开
            "APP权限未开启（如位置权限）",  # 原因2：系统权限限制
            "测量时手机未在旁",  # 原因3：手机不在身边
            "APP服务器维护",  # 原因4：服务端维护
        ],
        "solutions": [  # 解决方案清单
            "重新连接蓝牙",  # 方案1：对应原因1，重连蓝牙
            "在手机设置中为APP开启蓝牙和位置权限",  # 方案2：对应原因2，开权限
            "测量时保持手机在1米内",  # 方案3：对应原因3，手机近距陪伴
            "稍后重试或联系APP客服",  # 方案4：对应原因4，等待或转客服
        ],
    },
}  # 九大症状手册结束（graph_troubleshoot 决策图谱亦基于此构建）


# ── 嵌入工具（懒加载）──────────────────────────────────────────────────────────
_st_model = None  # sentence-transformers 模型单例缓存（None=未加载）


# ═══ 功能块：_load_st_model —— 惰性加载本地 BGE embedding 模型（失败则降级） ═══
def _load_st_model():
    """加载 sentence-transformers 本地模型 bge-small-zh-v1.5"""  # 原有 docstring
    global _st_model  # 声明修改模块级单例
    if _st_model is not None:  # 已加载过直接复用（避免重复加载耗时）
        return _st_model  # 返回缓存实例
    try:  # 模型加载可能失败（缺库/缺模型文件）
        from sentence_transformers import SentenceTransformer  # 延迟导入：用到才加载重依赖
        _st_model = SentenceTransformer(LOCAL_MODEL_PATH)  # 从本地路径加载 BGE 模型
        logger.info(f"已加载本地 embedding 模型: {LOCAL_MODEL_PATH}")  # 记录加载成功
    except Exception as e:  # 捕获一切加载异常
        logger.warning(f"加载本地模型失败: {e}，将降级使用关键词匹配")  # 记录失败与降级策略
        _st_model = None  # 保持 None，触发上层降级路径
    return _st_model  # 返回模型实例或 None


# ═══ 功能块：_embed_texts_dashscope —— DashScope 云端批量文本编码（建索引用） ═══
def _embed_texts_dashscope(texts: list) -> list:
    """调用 DashScope text-embedding-v3 对文本批量编码"""  # 原有 docstring
    client = _get_dashscope_client()  # 获取（惰性创建的）客户端单例
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)  # 一次请求批量编码全部文本
    vecs = []  # 归一化向量收集器
    for d in resp.data:  # 遍历返回的每条向量
        v = np.array(d.embedding, dtype="float32")  # 转 float32 数组（省内存且够精度）
        v = v / np.linalg.norm(v)  # L2 归一化：点积即余弦相似度
        vecs.append(v)  # 收纳归一化向量
    return vecs  # 返回向量列表


# ═══ 功能块：_embed_query_dashscope —— DashScope 云端单条查询编码（查询侧） ═══
def _embed_query_dashscope(text: str) -> np.ndarray:
    """调用 DashScope text-embedding-v3 对查询文本编码"""  # 原有 docstring
    client = _get_dashscope_client()  # 获取客户端单例
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])  # 单条文本编码
    vec = np.array(resp.data[0].embedding, dtype="float32")  # 取首条向量并转数组
    vec = vec / np.linalg.norm(vec)  # L2 归一化（与索引侧保持一致）
    return vec.reshape(1, -1)  # 变形为 (1, dim) 行向量，兼容矩阵运算


# ═══ 功能块：_build_tfidf_vectors —— 降级方案：字符级 TF 向量（零外部依赖） ═══
def _build_tfidf_vectors(texts: list) -> np.ndarray:
    """降级方案：构建简单的字符级 TF 向量（无外部模型依赖）"""  # 原有 docstring
    import re  # 局部导入正则（识别中文字符）
    # 构建字符 unigram + bigram 词典
    vocab = {}  # 词典：bigram/unigram → 维度下标
    for t in texts:  # 遍历每篇文本构建词典
        for i in range(len(t) - 1):  # 滑动窗口切 bigram
            bg = t[i:i + 2]  # 相邻两字符组成的二元组
            if re.match(r'[\u4e00-\u9fff]', bg):  # 仅保留中文二元组
                vocab.setdefault(bg, len(vocab))  # 新词登记维度下标
        for ch in t:  # 再遍历单字符
            if re.match(r'[\u4e00-\u9fff]', ch):  # 仅保留中文字符
                vocab.setdefault(ch, len(vocab))  # 新字登记维度下标

    vecs = []  # 向量收集器
    for t in texts:  # 第二遍：按词典统计词频生成向量
        v = np.zeros(len(vocab), dtype="float32")  # 全零向量（维度=词典大小）
        for i in range(len(t) - 1):  # 统计 bigram 词频
            bg = t[i:i + 2]  # 取二元组
            if bg in vocab:  # 命中词典
                v[vocab[bg]] += 1  # 对应维度计数 +1
        for ch in t:  # 统计 unigram 词频
            if ch in vocab:  # 命中词典
                v[vocab[ch]] += 1  # 对应维度计数 +1
        norm = np.linalg.norm(v)  # 计算向量 L2 范数
        if norm > 0:  # 非零向量才归一化（全零会除零）
            v = v / norm  # 归一化
        vecs.append(v)  # 收纳该文本向量
    return vecs  # 返回全部降级向量


# ═══ 功能块：_embed_query_local —— 本地模型编码查询；模型不可用返回 None 触发降级 ═══
def _embed_query_local(text: str) -> np.ndarray:
    """使用本地模型或降级 TF 对查询编码"""  # 原有 docstring
    model = _load_st_model()  # 惰性加载本地模型（可能返回 None）
    if model is not None:  # 本地模型可用
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)  # 编码并直接归一化
        return np.array(vec, dtype="float32")  # 统一转 float32 数组
    else:  # 本地模型不可用分支
        # 降级 TF 向量
        # 用与建索引时相同的 vocab 维度，但 vocab 不持久化，这里用动态匹配
        # 简化处理：直接用关键词匹配
        return None  # 由调用方处理 None


# ═══ 功能块：_embed_query —— 查询编码统一入口：按配置分流 云端/本地/降级 ═══
def _embed_query(text: str) -> np.ndarray:
    """根据配置选择 embedding 方式"""  # 原有 docstring
    if EMBED_PROVIDER == "dashscope":  # 配置为云端
        return _embed_query_dashscope(text)  # 走 DashScope 编码
    else:  # 配置为本地
        vec = _embed_query_local(text)  # 走本地模型编码
        if vec is not None:  # 本地编码成功
            return vec  # 返回本地向量
        # 降级：用 FAQ 主题关键词匹配
        return None  # 返回 None，调用方据此走关键词降级检索


# ── 混合检索（BM25 + BGE + RRF + Reranker，懒加载）─────────────────────────────
_hybrid_retriever = None  # 混合检索器单例缓存（懒加载）


# ═══ 功能块：_build_hybrid_docs —— 把 FAQ/产品/故障三类知识拼成统一检索语料 ═══
def _build_hybrid_docs() -> list[dict]:
    """构建混合检索语料：FAQ + 产品规格 + 故障症状 + 测量指南。
    产品、故障条目加入索引，显著提升型号与专有名词召回率。"""  # 原有 docstring
    docs = [{"id": f["id"], "topic": f["topic"], "content": f["content"], "type": "faq"}  # FAQ 条目直接映射为文档
            for f in FAQ_KNOWLEDGE_BASE]  # 列表推导：20 条 FAQ 全部入库
    for name, info in PRODUCT_MAP.items():  # 遍历 6 个产品型号
        docs.append({  # 每个型号拼成一篇"规格说明书"文档
            "id": f"PROD-{info['code']}", "topic": name,  # 文档 id 加 PROD- 前缀便于溯源
            "content": (f"产品代码 {info['code']}，{info['brand']}，{info['type']}，"  # 拼接：代码/品牌/类型
                        f"适用上臂周长 {info['cuff_range']}，供电 {info['power']}，"  # 拼接：袖带/供电
                        f"存储 {info['memory']}，功能 {info['features']}"),  # 拼接：存储/功能
            "type": "product",  # 标记来源类型：产品规格
        })
    for symptom, info in TROUBLESHOOT_MAP.items():  # 遍历 9 个故障症状
        docs.append({  # 每个症状拼成一篇"排查手册"文档
            "id": f"TS-{symptom}", "topic": f"故障：{symptom}",  # 文档 id 加 TS- 前缀
            "content": ("可能原因：" + "；".join(info["possible_causes"]) +  # 拼接：原因列表
                        "。解决方案：" + "；".join(info["solutions"])),  # 拼接：方案列表
            "type": "troubleshoot",  # 标记来源类型：故障排查
        })
    return docs  # 返回统一格式的语料列表（id/topic/content/type 四字段）


# ═══ 功能块：_load_hybrid —— 构建或复用 BM25+BGE 混合检索器（全局单例） ═══
def _load_hybrid():
    """构建（或复用）BM25+BGE 混合检索器"""  # 原有 docstring
    global _hybrid_retriever  # 声明修改模块级单例
    if _hybrid_retriever is not None:  # 已构建过
        return _hybrid_retriever  # 直接复用（索引构建昂贵，只做一次）
    logger.info("正在构建混合检索索引（BM25 + BGE + RRF）...")  # 记录构建开始
    from hybrid_search import HybridRetriever  # 延迟导入：检索器实现位于同目录 hybrid_search
    _hybrid_retriever = HybridRetriever(_build_hybrid_docs())  # 用三源语料构建混合检索器
    return _hybrid_retriever  # 返回构建好的单例


# ═══ 功能块：_load_rag —— 向后兼容入口：老接口现转发到混合检索 ═══
def _load_rag():
    """向后兼容入口：老代码 serve.py 调用的预加载，现指向混合检索"""  # 原有 docstring
    _load_hybrid()  # 直接转发（保持旧函数名可用）


# ═══ 功能块：tool_faq_search —— Agent 工具①：知识库混合检索（含三级降级） ═══
def tool_faq_search(query: str, top_k: int = 5) -> str:
    """在血压计 FAQ/产品/故障知识库中做混合检索（BM25+BGE→RRF→Reranker），返回最相关内容"""  # 原有 docstring
    try:  # 最外层兜底：任何异常都转为错误字符串返回给 LLM
        if EMBED_PROVIDER != "local" or _load_st_model() is not None:  # 云端模式或本地模型就绪才走混合检索
            try:  # 混合检索自身的异常单独捕获（便于降级）
                retriever = _load_hybrid()  # 获取（惰性构建的）检索器单例
                hits = retriever.search(query, top_k)  # 执行混合检索，返回 Top-K 命中
                if hits:  # 有命中结果
                    results = []  # 格式化结果收集器
                    for rank, h in enumerate(hits, 1):  # 按相关度顺序编号
                        source = {"faq": f"FAQ-{h.get('id','')}",  # faq 类型 → FAQ-编号
                                  "product": "产品规格库",  # product 类型 → 产品规格库
                                  "troubleshoot": "故障排查手册"}.get(h.get("type"), h.get("type"))  # troubleshoot 类型 → 故障手册（其余原样）
                        results.append(  # 拼一条带来源与融合分的检索结果
                            f"[{rank}] 来源：{source} 主题：{h.get('topic','')} "  # 排名 + 来源 + 主题
                            f"(融合分:{h.get('score', 0):.3f})\n{h['content']}"  # RRF 融合分 + 原文内容
                        )
                    if results:  # 格式化列表非空
                        return "\n\n".join(results)  # 空行分隔拼接返回
            except Exception as e:  # 混合检索失败（索引构建/模型加载等）
                logger.warning(f"混合检索失败({e})，降级关键词匹配")  # 记录降级原因

        # 降级：关键词匹配
        return _keyword_search(query, top_k)  # 二级降级：纯关键词重叠度匹配

    except Exception as e:  # 最外层兜底捕获
        return f"faq_search 执行出错: {e}"  # 返回错误描述而非抛出（工具契约）


# ═══ 功能块：_keyword_search —— 降级检索：FAQ 关键词重叠度打分（零依赖） ═══
def _keyword_search(query: str, top_k: int) -> str:
    """降级关键词匹配：基于字符级集合重叠度打分（修正：原整串集合导致中文几乎无法命中）"""  # 原有 docstring
    import re  # 局部导入正则
    # 提取查询中的中文字符集合（字符级：整串集合在中文场景几乎不可能完全相等）
    query_chars = set("".join(re.findall(r'[\u4e00-\u9fff]+', query)))  # 查询中文字符集
    MIN_OVERLAP = 0.3  # 最小重叠率阈值：低于该值视为不相关，避免垃圾查询返回噪声
    scored = []  # (得分, FAQ条目) 收集器
    for f in FAQ_KNOWLEDGE_BASE:  # 遍历 20 条 FAQ 逐一打分
        content = f["topic"] + " " + f["content"]  # 主题 + 正文拼接为匹配文本
        content_chars = set("".join(re.findall(r'[\u4e00-\u9fff]+', content)))  # 该条 FAQ 的中文字符集
        # 字符级 Jaccard 重叠率
        overlap = len(query_chars & content_chars)  # 查询字符与文档字符的交集大小
        score = overlap / max(len(query_chars), 1) if query_chars else 0  # 交集占查询字符比例（除零保护）
        if score < MIN_OVERLAP:  # 重叠率过低：跳过（视为不相关）
            continue
        scored.append((score, f))  # 记录该条得分

    scored.sort(key=lambda x: x[0], reverse=True)  # 按得分降序排序
    results = []  # 格式化结果收集器
    for rank, (score, f) in enumerate(scored[:top_k], 1):  # 取前 top_k 条
        results.append(  # 拼一条检索结果（格式与混合检索保持一致）
            f"[{rank}] 来源：FAQ-{f['id']} 主题：{f['topic']} (关键词匹配:{score:.3f})\n{f['content']}"  # 排名 + 来源 + 得分 + 原文
        )
    return "\n\n".join(results) if results else "未检索到相关内容"  # 有结果拼接返回，否则提示未命中


# ── 产品查询工具 ──────────────────────────────────────────────────────────────

# ═══ 功能块：tool_product_lookup —— Agent 工具②：型号名 → 产品规格（精确→模糊→提示） ═══
def tool_product_lookup(model: str) -> str:
    """将血压计型号名称转换为产品规格信息"""  # 原有 docstring
    model = model.strip()  # 去除首尾空白（LLM 传参可能带空格）
    # 精确匹配
    if model in PRODUCT_MAP:  # 完整型号名直接命中
        return _format_product(model, PRODUCT_MAP[model])  # 格式化返回规格
    # 模糊匹配
    candidates = [k for k in PRODUCT_MAP if model in k or k in model]  # 双向子串匹配找相似型号
    if candidates:  # 有相似候选
        return "未精确匹配，相似型号：\n" + "\n\n".join(  # 提示未精确命中并列出候选
            _format_product(k, PRODUCT_MAP[k]) for k in candidates  # 候选型号逐一格式化
        )
    supported = "、".join(PRODUCT_MAP.keys())  # 拼接全部支持的型号清单
    return f"未找到型号 '{model}'，当前支持：{supported}"  # 兜底：告知支持范围引导重试


# ═══ 功能块：_format_product —— 产品规格 dict → 消费者可读多行文本 ═══
def _format_product(name: str, info: dict) -> str:
    """将产品规格字典格式化为消费者可读的文本（供 LLM 与前端直接展示）"""  # 原有 docstring
    return (  # 隐式拼接的多行格式化文本
        f"型号：{name}\n"  # 首行：完整型号名
        f"  产品代码：{info['code']}\n"  # 型号代码
        f"  品牌：{info['brand']}\n"  # 品牌
        f"  类型：{info['type']}\n"  # 设备类型
        f"  袖带适用范围：{info['cuff_range']}\n"  # 袖带臂围
        f"  供电方式：{info['power']}\n"  # 供电方式
        f"  存储容量：{info['memory']}\n"  # 存储容量
        f"  主要功能：{info['features']}"  # 特色功能（末行无换行符）
    )


# ── 计算器工具 ────────────────────────────────────────────────────────────────

# 白名单命名空间：仅暴露 math 公共函数与内置数值函数，屏蔽 __builtins__ 防止任意代码执行
_SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}  # 提取 math 公共函数（过滤下划线私有名）
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})  # 补充五个常用内置数值函数

# ═══ 功能块：tool_calculator —— Agent 工具③：沙箱化的数学表达式计算器 ═══
def tool_calculator(expr: str) -> str:
    """安全计算数学表达式，支持四则运算和 math 模块函数"""  # 原有 docstring
    try:  # 表达式可能非法（语法错/除零/未知函数）
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307  # 沙箱 eval：清空内置 + 白名单命名空间
        return str(round(float(result), 6))  # 转 float 四舍五入到 6 位小数再转字符串
    except Exception as e:  # 捕获计算异常
        return f"计算出错: {e}，表达式: {expr}"  # 返回错误描述（含原表达式便于 LLM 自纠错）


# ── 测量指南工具（结构化数据） ─────────────────────────────────────────────────

# ═══ 功能块：tool_measurement_guide —— Agent 工具④：八大主题测量指南查表 ═══
def tool_measurement_guide(topic: str) -> str:
    """根据主题返回血压测量标准操作指南"""  # 原有 docstring
    topic = topic.strip()  # 去除首尾空白
    guides = {  # 主题 → 指南文本的静态映射表（隐式字符串拼接成多行）
        "测量前准备": (  # 主题1：测量前的行为准备
            "【血压测量前准备指南】\n"  # 标题行
            "1. 测量前30分钟内禁止吸烟、饮酒、喝咖啡或浓茶\n"  # 条目1：刺激物禁忌
            "2. 避免剧烈运动，安静环境静坐休息5-10分钟\n"  # 条目2：静息要求
            "3. 排空膀胱，避免憋尿\n"  # 条目3：排尿
            "4. 保持情绪平稳，避免紧张焦虑\n"  # 条目4：情绪管理
            "5. 测量前避免洗热水澡（会使血管扩张血压下降）\n"  # 条目5：热水澡影响
            "6. 建议在每天固定时间测量，便于趋势对比"  # 条目6：固定时间
        ),
        "袖带绑法": (  # 主题2：袖带佩戴规范
            "【袖带正确绑法指南】\n"  # 标题行
            "1. 裸露上臂或仅穿薄衣，避免卷袖过紧压迫血管\n"  # 条目1：着装要求
            "2. 将袖带缠绕在上臂，下缘距肘窝2-3cm（约两横指）\n"  # 条目2：绑带位置
            "3. 袖带气导管位于手臂内侧中轴线方向\n"  # 条目3：气管朝向
            "4. 袖带松紧适中，以能伸入1-2根手指为宜\n"  # 条目4：松紧标准
            "5. 袖带不可压住肘窝，也不可过松或过紧\n"  # 条目5：位置禁忌
            "6. 确保袖带标签/箭头位于肱动脉上方"  # 条目6：对准肱动脉
        ),
        "测量姿势": (  # 主题3：坐姿与手臂摆放
            "【正确测量姿势指南】\n"  # 标题行
            "1. 坐于有靠背的椅子上，背部靠椅背\n"  # 条目1：靠背坐姿
            "2. 双脚平放地面，不可交叉双腿\n"  # 条目2：双脚平放
            "3. 手臂自然放在桌面上，掌心向上\n"  # 条目3：手臂摆放
            "4. 袖带中心与心脏（第四肋间腋中线）同高水平\n"  # 条目4：心脏同高
            "5. 身体放松不可前倾后仰\n"  # 条目5：躯干稳定
            "6. 测量过程中不要说话、不要移动手臂"  # 条目6：测量中禁动
        ),
        "测量次数": (  # 主题4：次数/间隔/取值规则
            "【测量次数与取值指南】\n"  # 标题行
            "1. 每次建议连续测量2-3次\n"  # 条目1：建议次数
            "2. 每次测量间隔1-2分钟，待手臂血液循环恢复\n"  # 条目2：间隔要求
            "3. 取后两次测量的平均值作为最终结果\n"  # 条目3：取值规则
            "4. 首次测量值通常偏高（紧张/加压刺激），可舍去\n"  # 条目4：首测偏高说明
            "5. 每日早晚各测一次（晨起服药前、晚睡前）\n"  # 条目5：早晚监测
            "6. 连续监测7天，数据更可靠"  # 条目6：连续监测周期
        ),
        "左右手选择": (  # 主题5：测量手臂的选择原则
            "【测量手臂选择指南】\n"  # 标题行
            "1. 首次测量时左右手各测一次\n"  # 条目1：首测双臂对比
            "2. 选择血压值较高的一侧作为固定测量手臂\n"  # 条目2：固定较高侧
            "3. 正常左右手差异5-10mmHg属正常\n"  # 条目3：正常差异范围
            "4. 若双臂收缩压差异>20mmHg，需就医排查\n"  # 条目4：就医阈值
            "5. 偏瘫患者测量健侧手臂\n"  # 条目5：偏瘫人群
            "6. 乳腺癌术后患者测量健侧手臂"  # 条目6：术后人群
        ),
        "血压分级": (  # 主题6：血压分级标准（指南查表）
            "【血压分级标准指南（中国高血压防治指南2018）】\n"  # 标题行（注明出处）
            "正常血压：收缩压<120 且 舒张压<80 mmHg\n"  # 正常血压阈值
            "正常高值：收缩压120-139 或 舒张压80-89 mmHg\n"  # 正常高值区间
            "1级高血压（轻度）：收缩压140-159 或 舒张压90-99 mmHg\n"  # 1级高血压
            "2级高血压（中度）：收缩压160-179 或 舒张压100-109 mmHg\n"  # 2级高血压
            "3级高血压（重度）：收缩压≥180 或 舒张压≥110 mmHg\n"  # 3级高血压
            "单纯收缩期高血压：收缩压≥140 且 舒张压<90 mmHg\n"  # 单纯收缩期型
            "注意：家庭自测血压≥135/85mmHg即应考虑高血压"  # 家庭自测特别阈值
        ),
        "平均动脉压": (  # 主题7：MAP 计算公式与临床范围
            "【平均动脉压（MAP）计算指南】\n"  # 标题行
            "平均动脉压 = 舒张压 + 1/3 × (收缩压 - 舒张压)\n"  # 计算公式
            "正常范围：70-100 mmHg\n"  # 正常范围
            "MAP < 65 mmHg：组织灌注不足风险\n"  # 低 MAP 风险
            "MAP > 110 mmHg：高血压靶器官损害风险\n"  # 高 MAP 风险
            "示例：收缩压120，舒张压80 → MAP = 80 + 1/3×(120-80) = 93.3 mmHg\n"  # 计算示例
            "可使用 calculator 工具计算具体数值"  # 引导调用计算器工具
        ),
        "脉压差": (  # 主题8：脉压差公式与临床解读
            "【脉压差（脉搏压）指南】\n"  # 标题行
            "脉压差 = 收缩压 - 舒张压\n"  # 计算公式
            "正常范围：30-40 mmHg\n"  # 正常范围
            "脉压差增大（>60mmHg）：常见于主动脉硬化、甲亢、贫血、主动脉瓣关闭不全\n"  # 增大的病因
            "脉压差减小（<25mmHg）：常见于主动脉瓣狭窄、心包积液、休克\n"  # 减小的病因
            "老年人脉压差增大是心血管事件的重要预测因子"  # 老年人群意义
        ),
    }  # 八大主题指南表结束

    # 精确匹配
    if topic in guides:  # 主题名完全命中
        return guides[topic]  # 直接返回对应指南全文

    # 模糊匹配主题
    candidates = [k for k in guides if topic in k or k in topic]  # 双向子串匹配找相近主题
    if candidates:  # 有相近主题
        return "匹配到以下主题指南：\n" + "\n\n".join(guides[k] for k in candidates)  # 提示并列出全部候选指南

    return (  # 兜底：未知主题
        f"未找到主题 '{topic}'，当前支持的主题：\n"  # 提示未命中
        + "\n".join(f"  - {k}" for k in guides)  # 列出全部支持主题供 LLM 换词重试
    )


# ── 故障排查工具 ─────────────────────────────────────────────────────────────

# ═══ 功能块：tool_troubleshoot —— Agent 工具⑤：症状 → 原因+方案（精确→模糊→提示） ═══
def tool_troubleshoot(symptom: str) -> str:
    """根据故障症状返回可能原因和解决方案"""  # 原有 docstring
    symptom = symptom.strip()  # 去除首尾空白

    # 精确匹配
    if symptom in TROUBLESHOOT_MAP:  # 症状名完全命中
        return _format_troubleshoot(symptom, TROUBLESHOOT_MAP[symptom])  # 格式化返回该症状手册

    # 模糊匹配
    candidates = [k for k in TROUBLESHOOT_MAP if symptom in k or k in symptom]  # 双向子串匹配找相近症状
    if candidates:  # 有相近症状
        results = []  # 结果收集器
        for k in candidates:  # 遍历候选症状
            results.append(_format_troubleshoot(k, TROUBLESHOOT_MAP[k]))  # 逐一格式化
        return "匹配到以下故障：\n" + "\n\n".join(results)  # 提示并列出全部候选

    supported = "、".join(TROUBLESHOOT_MAP.keys())  # 拼接全部可排查症状清单
    return f"未找到故障 '{symptom}'，当前支持排查的故障：{supported}"  # 兜底：告知支持范围


# ═══ 功能块：_format_troubleshoot —— 排查 dict（原因+方案）→ 编号列表文本 ═══
def _format_troubleshoot(symptom: str, info: dict) -> str:
    """将故障排查字典（可能原因 + 解决方案）格式化编号列表文本"""  # 原有 docstring
    causes = "\n".join(f"  {i}. {c}" for i, c in enumerate(info["possible_causes"], 1))  # 原因列表逐条编号
    solutions = "\n".join(f"  {i}. {s}" for i, s in enumerate(info["solutions"], 1))  # 方案列表逐条编号
    return (  # 拼装为"故障→原因→方案"三段式文本
        f"【故障：{symptom}】\n"  # 首行：故障标题
        f"可能原因：\n{causes}\n"  # 原因段
        f"解决方案：\n{solutions}"  # 方案段（末行无换行符）
    )


# ── 图谱排查工具（决策树/Graph 增强：结构化逻辑 + 语义检索联合推演） ──────────

# ═══ 功能块：tool_graph_troubleshoot —— Agent 工具⑥：决策图谱排查（失败降级线性） ═══
def tool_graph_troubleshoot(symptom: str) -> str:
    """基于故障排查决策图谱，输出按优先级排序的排查决策路径（含多跳深入排查）"""  # 原有 docstring
    try:  # 图谱模块可能不可用（依赖缺失/初始化失败）
        from troubleshoot_graph import graph_resolve, format_graph_result  # 延迟导入：图谱推演与结果格式化
        result = graph_resolve(symptom.strip())  # 在决策图上推演出分步排查路径
        return format_graph_result(result)  # 格式化为可读的分步指引文本
    except Exception as e:  # 图谱推演失败
        # 图谱不可用时降级到原线性排查
        logger.warning(f"图谱排查失败({e})，降级线性排查")  # 记录降级原因
        return tool_troubleshoot(symptom)  # 降级：回退到线性排查手册


# ── 统一工具注册表 ─────────────────────────────────────────────────────────────

TOOLS_MAP: dict[str, Any] = {  # 工具名 → 实现函数的注册表：手写版 ReAct 按名分发调用
    "faq_search":          tool_faq_search,  # 知识库混合检索
    "product_lookup":      tool_product_lookup,  # 型号 → 产品规格
    "calculator":          tool_calculator,  # 沙箱数学计算器
    "measurement_guide":   tool_measurement_guide,  # 八大主题测量指南
    "troubleshoot":        tool_troubleshoot,  # 症状 → 原因+方案（线性）
    "graph_troubleshoot":  tool_graph_troubleshoot,  # 症状 → 分步排查路径（图谱）
}

# Function Calling 版所需的 JSON Schema 描述
TOOLS_SCHEMA = [  # 6 个工具的 OpenAI Function Calling Schema 列表：传给 LLM 的"工具签名"，LLM 据此自主决定调用时机与参数填法
    {  # ── 工具① faq_search：知识库混合检索 ──
        "type": "function",  # Schema 类型：函数调用类工具
        "function": {  # 函数定义体：名称 + 用途描述 + 参数约束
            "name": "faq_search",  # 函数名：与 TOOLS_MAP 的 key 一致，运行时按此名分发到实现函数
            "description": "在血压计知识库（20条FAQ+6个产品型号规格+9种故障排查手册）中做混合检索（BM25关键词+语义向量融合+重排），适合查询定性描述、操作步骤、注意事项、型号参数等文本内容，对型号（如HEM-7156）和专有名词召回更准",  # 用途描述：LLM 依据此判断何时调用（定性问题查询 / 型号专有名词精确召回）
            "parameters": {  # 参数 JSON Schema：约束 LLM 调用时需填的参数
                "type": "object",  # 参数整体类型：键值对对象
                "properties": {  # 各参数的类型与含义定义
                    "query": {"type": "string", "description": "检索问题，尽量具体，如'血压计怎么绑袖带'"},  # 必填参数：检索问题（越具体召回越准）
                    "top_k": {"type": "integer", "description": "返回结果数量，默认5", "default": 5},  # 可选参数：返回条数，带默认值 5
                },
                "required": ["query"],  # 必填参数清单：仅 query 必填，top_k 可省略
            },
        },
    },
    {  # ── 工具② product_lookup：型号 → 产品规格 ──
        "type": "function",  # Schema 类型：函数调用
        "function": {  # 函数定义体
            "name": "product_lookup",  # 函数名：与 TOOLS_MAP 的 key 一致
            "description": "将血压计型号名称（如'欧姆龙HEM-7156'）转换为产品规格信息（品牌/类型/袖带范围/供电/存储/功能），在查询具体型号参数前必须先用此工具",  # 用途描述：末尾强调"必须先用"，引导 LLM 建立先查规格再回答的调用顺序
            "parameters": {  # 参数 schema
                "type": "object",  # 参数对象
                "properties": {  # 参数定义
                    "model": {"type": "string", "description": "血压计型号名称，如'欧姆龙HEM-7156'"},  # 必填参数：型号名称（键名需与 PRODUCT_MAP 匹配）
                },
                "required": ["model"],  # model 必填
            },
        },
    },
    {  # ── 工具③ calculator：沙箱数学计算器 ──
        "type": "function",  # Schema 类型：函数调用
        "function": {  # 函数定义体
            "name": "calculator",  # 函数名：与 TOOLS_MAP 的 key 一致
            "description": "安全计算数学表达式，支持加减乘除、幂运算、math模块函数（sqrt/log/pow等），用于血压相关计算如平均动脉压MAP、脉压差、血压变化率等",  # 用途描述：列明支持的运算范围与典型计算场景，避免 LLM 心算数字
            "parameters": {  # 参数 schema
                "type": "object",  # 参数对象
                "properties": {  # 参数定义
                    "expr": {"type": "string", "description": "数学表达式，如 '80 + (120-80)/3' 计算平均动脉压"},  # 必填参数：表达式字符串（示例即 MAP 公式）
                },
                "required": ["expr"],  # expr 必填
            },
        },
    },
    {  # ── 工具④ measurement_guide：八大主题测量指南 ──
        "type": "function",  # Schema 类型：函数调用
        "function": {  # 函数定义体
            "name": "measurement_guide",  # 函数名：与 TOOLS_MAP 的 key 一致
            "description": "根据主题返回血压测量标准操作指南，支持主题：测量前准备/袖带绑法/测量姿势/测量次数/左右手选择/血压分级/平均动脉压/脉压差，适合查询操作规范和医学标准",  # 用途描述：枚举八大主题，供 LLM 对号入座选择入参
            "parameters": {  # 参数 schema
                "type": "object",  # 参数对象
                "properties": {  # 参数定义
                    "topic": {"type": "string", "description": "指南主题，如'袖带绑法'"},  # 必填参数：指南主题名
                },
                "required": ["topic"],  # topic 必填
            },
        },
    },
    {  # ── 工具⑤ troubleshoot：线性故障排查手册 ──
        "type": "function",  # Schema 类型：函数调用
        "function": {  # 函数定义体
            "name": "troubleshoot",  # 函数名：与 TOOLS_MAP 的 key 一致
            "description": "根据故障症状返回可能原因和解决方案，支持故障：测不出值/测量值偏高/测量值偏低/数值波动大/无法开机/屏幕不显示/漏气/蓝牙连不上/APP数据不同步，适合排查设备使用问题",  # 用途描述：枚举 9 种可排查症状，方便 LLM 判断症状是否在支持范围
            "parameters": {  # 参数 schema
                "type": "object",  # 参数对象
                "properties": {  # 参数定义
                    "symptom": {"type": "string", "description": "故障症状描述，如'测量值偏高'"},  # 必填参数：症状描述
                },
                "required": ["symptom"],  # symptom 必填
            },
        },
    },
    {  # ── 工具⑥ graph_troubleshoot：决策图谱分步排查 ──
        "type": "function",  # Schema 类型：函数调用
        "function": {  # 函数定义体
            "name": "graph_troubleshoot",  # 函数名：与 TOOLS_MAP 的 key 一致
            "description": "基于故障排查决策图谱输出按优先级排序的排查决策路径（先易后难，含多跳深入排查：如'测不出值'会深入排查漏气/电池，'蓝牙连不上'会联动APP数据不同步）。适合用户症状需要分步排查引导的场景，优先于 troubleshoot 使用",  # 用途描述：说明先易后难排序与多跳联动能力，并声明优先于 troubleshoot 使用
            "parameters": {  # 参数 schema
                "type": "object",  # 参数对象
                "properties": {  # 参数定义
                    "symptom": {"type": "string", "description": "故障症状描述，如'测不出值'、'蓝牙连不上'"},  # 必填参数：症状描述（给出两个多跳场景示例）
                },
                "required": ["symptom"],  # symptom 必填
            },
        },
    },
]  # 6 个 schema 与 TOOLS_MAP 一一对应；react_function_calling.py 将此列表直接传给 LLM 的 tools 参数

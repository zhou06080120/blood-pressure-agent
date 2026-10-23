"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent —— 血压计客服

使用方式：
  # 单轮（原有用法）
  python agent.py
  python agent.py --mode manual   --question "血压计袖带怎么绑？"
  python agent.py --mode fc       --question "测出来偏高怎么办？"

  # 多轮交互模式（新增）
  python agent.py --interactive
  python agent.py --interactive --mode manual

  # 多轮交互中可输入命令：
  /history  查看当前对话历史
  /clear    清空对话历史，开始新话题
  /quit     退出

环境变量：
  DASHSCOPE_API_KEY  必填（两版统一经 llm_gateway 主链路接入）
  AGENT_MODEL        默认 qwen-max；备链路用 BP_LLM_FALLBACK_* 配置
  BP_EMBED_PROVIDER  local（默认）/ dashscope
"""

# ═══ 导入区：os 读环境变量，argparse 解析命令行参数 ═══
import os  # 读环境变量（AGENT_MODEL 等）与设置 KMP 兼容开关
import argparse  # 解析 --mode/--question/--max_steps/--interactive 参数

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许系统内存在多份 OpenMP 运行库，避免 numpy/faiss 初始化时崩溃

DEFAULT_QUESTION = "血压计袖带应该怎么绑？测出来的值偏高怎么办？"  # 不传 --question 时的默认演示问题


# ═══ 功能块：单轮模式 —— 跑一次 ReAct 并逐步打印全过程 ═══
def run_single(args):
    """单轮模式：运行一次并打印"""
    if args.mode == "manual":  # manual = 手写 Prompt 解析版
        from react_manual import run_and_print  # 延迟导入对应实现（只加载被选中的那版，加快启动）
    else:  # 其余情况走 Function Calling 版
        from react_function_calling import run_and_print
    run_and_print(args.question, args.max_steps)  # 执行单轮问答并打印 Thought/Action/Observation/Final


# ═══ 功能块：多轮交互模式 —— 循环提问、累积历史、支持 /history /clear /quit 命令 ═══
def run_interactive(args):
    """多轮交互模式：循环提问，累积对话历史"""
    if args.mode == "manual":  # 按命令行模式选择实现
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    history = []  # 跨轮只存 user 问题 + assistant 最终回答

    print("\n" + "=" * 60)  # 打印横幅分隔线
    print("🩺 血压计客服 Agent — 多轮对话模式")  # 标题
    print(f"   实现: {args.mode}  模型: {os.getenv('AGENT_MODEL','qwen-max')}")  # 显示当前实现方式与模型名
    print("   命令: /history 查看历史  /clear 清空历史  /quit 退出")  # 提示可用命令
    print("=" * 60)  # 分隔线收尾
    print("您好！我是血压计客服助手，可以解答袖带绑法、测量姿势、")  # 欢迎语第一行
    print("正常范围、故障排查、蓝牙连接等问题。请问有什么可以帮您？\n")  # 欢迎语第二行

    while True:  # 主交互循环：不断读取并处理用户输入
        try:  # 捕获输入流关闭/用户打断
            question = input("\n👤 您: ").strip()  # 读入一行并去掉首尾空白
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z/Ctrl+D 或 Ctrl+C
            print("\n再见！")  # 告别语
            break  # 退出交互循环

        if not question:  # 空输入
            continue  # 忽略并继续读取
        if question in ("/quit", "/exit", "/q"):  # 退出命令（三种别名）
            print("再见！")  # 告别语
            break
        if question == "/history":  # 查看历史命令
            if not history:  # 还没有任何对话记录
                print("（暂无对话历史）")  # 提示为空
            else:
                print(f"\n📜 对话历史（{len(history)}条消息，{len(history)//2}轮）：")  # 打印消息数与轮数（2 条消息 = 1 轮）
                for i, m in enumerate(history):  # 逐条遍历历史
                    role = "👤 您" if m["role"] == "user" else "🤖 客服"  # 按角色选择显示图标
                    print(f"  [{i+1}] {role}: {m['content'][:120]}{'...' if len(m['content'])>120 else ''}")  # 超过 120 字截断显示
            continue
        if question == "/clear":  # 清空历史命令
            history.clear()  # 原地清空列表（引用不变）
            print("✅ 对话历史已清空，开始新话题。")  # 提示已清空
            continue

        # 运行 ReAct，带入历史
        final_answer = None  # 先置空，用于捕获本轮最终回答
        for step_data in run_and_print(question, max_steps=args.max_steps, history=history):  # 带历史执行并逐步打印
            if step_data.get("type") == "final":  # 捕获 final 类型事件
                final_answer = step_data.get("answer", "")  # 记录最终答案文本

        # 回写历史：user 问题 + assistant 最终回答
        if final_answer:  # 只有成功产出答案才回写历史
            history.append({"role": "user", "content": question})  # 追加本轮用户问题
            history.append({"role": "assistant", "content": final_answer})  # 追加助手最终回答（不含中间步骤）
            # 控制历史长度，避免上下文过长
            if len(history) > 20:  # 超过 20 条消息
                history[:] = history[-20:]  # 原地截断，只保留最近 20 条
            print(f"\nℹ️  当前已记录 {len(history)//2} 轮对话")  # 提示当前已记录轮数


if __name__ == "__main__":  # 直接运行本文件才解析参数并执行
    parser = argparse.ArgumentParser(description="ReAct 血压计客服 Agent")  # 创建参数解析器
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",  # 实现模式：manual（默认）或 fc
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION,  # 单轮问题，缺省用默认演示题
                        help="单轮模式下的问题")
    parser.add_argument("--max_steps", type=int, default=10)  # ReAct 循环最大步数
    parser.add_argument("--interactive", "-i", action="store_true",  # 开关参数：进入多轮交互模式
                        help="多轮交互模式，循环提问并累积对话历史")
    args = parser.parse_args()  # 解析命令行参数

    if args.interactive:  # 带 -i 时进入多轮模式
        run_interactive(args)
    else:  # 否则执行单轮问答
        run_single(args)

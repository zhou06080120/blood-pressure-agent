# 公网部署手册（blood-pressure-agent）

> Render / UptimeRobot 的通用操作步骤与状态页搭建见姊妹项目
> [bp-monitor-service-agent/deploy/DEPLOY.md](../bp_monitor_service_agent/deploy/DEPLOY.md)，
> 本手册只写本项目差异。

## 免费额度策略（重要）

Render 免费版每月 **750 实例小时 ≈ 只够一个服务常驻**。三个项目的分配：

| 服务 | 策略 | 效果 |
|---|---|---|
| bp-monitor-service-agent | UptimeRobot 5 分钟心跳保活 | 常驻，秒开（主 Demo） |
| **本项目** | **不配心跳** | 休眠唤醒式，首次打开冷启动约 1 分钟 |
| blodd-answer-piper | 不配心跳 | 同上（模型加载更久） |

需要向 HR / 面试官重点演示本项目时：临时把本服务加进 UptimeRobot 监控（5 分钟内唤醒），**演示完记得删除**，否则三个服务常驻会超出免费时长。

## 部署步骤（约 10 分钟）

1. render.com → **New → Blueprint** → 选本仓库 → 自动识别 `render.yaml` 创建 `blood-pressure-agent`
2. 弹窗填三个环境变量（只存 Render，不进仓库）：
   - `DASHSCOPE_API_KEY`：主链路 qwen（DashScope OpenAI 兼容端点）
   - `BP_LLM_FALLBACK_API_KEY`：可选，备链路 Key（render.yaml 已预设 `https://api.deepseek.com`，主链路抖动自动切换）
   - `BP_ADMIN_KEYS`：**必填**随机长串（保护清缓存等管理端点）
3. build 完成后验证：`curl https://<服务名>.onrender.com/health` + 打开根路径 Demo 页

## 无 Key 时的降级行为（如实告知访客）

`DASHSCOPE_API_KEY` 留空服务照常启动：检索（TF 向量 + BM25）、工具调用编排、安全防线、限流鉴权全部可演示；仅 Agent 对话生成不可用。镜像默认轻量形态（不带 BGE 目录），`BP_EMBED_PROVIDER=local` 自动降级 TF 向量。

## README 接线

部署后替换 `README.md`「在线体验」一节的 `{YOUR-APP}.onrender.com` 占位符。

## 国内访问

与主项目相同：上线后用手机蜂窝网络实测；不稳则切 Zeabur / 轻量云，配置同 `render.yaml`。

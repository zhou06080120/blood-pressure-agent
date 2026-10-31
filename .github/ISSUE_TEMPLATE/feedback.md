name: badcase 反馈
about: Demo 回答错误 / 工具调用异常时提交，帮助定位检索与推理链路问题
title: "[badcase] "
labels: feedback
body:
  - type: textarea
    id: question
    attributes:
      label: 你的提问
      description: 在 Demo 里输入的原始问题（多跳问题请写完整，如「袖带怎么绑 + 测量值偏高怎么办」）
    validations:
      required: true
  - type: textarea
    id: answer
    attributes:
      label: Agent 的实际回答 / Thought-Action 链
      description: 粘贴 Demo 页面回答原文；如可见推理链（Thought → Action → Observation）请一并粘贴
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: 期望的正确回答（可选）
    validations:
      required: false

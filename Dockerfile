# ═════════════════════════════════════════════════════════════════
# 多阶段构建：builder 装依赖 → runtime 只带运行时产物
#
# 默认形态：轻量镜像（openai/fastapi/uvicorn/numpy/faiss-cpu），
#   无本地 BGE 模型时检索自动降级 TF 向量 + BM25，零外部依赖即可起服务。
# 完整形态：--build-arg INSTALL_EMBEDDINGS=1 追加 sentence-transformers，
#   配合 compose 挂载本地 BGE 模型目录启用 Dense 检索（见 docker-compose.yml）。
#
# 构建：  docker build -t blood-pressure-agent .
# 运行：  docker run -p 8000:8000 blood-pressure-agent
# ═════════════════════════════════════════════════════════════════

# ── 阶段 1：安装依赖（依赖层缓存：仅 pyproject 变化时重建） ───────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
# 虚包安装：只落地 dependencies 声明的第三方包，不打包应用代码
RUN pip install --no-cache-dir --target=/deps .

# 可选：完整 embedding 栈（sentence-transformers + torch，约 +1.5GB）
ARG INSTALL_EMBEDDINGS=0
RUN if [ "$INSTALL_EMBEDDINGS" = "1" ]; then \
        pip install --no-cache-dir --target=/deps sentence-transformers; \
    fi

# ── 阶段 2：运行时镜像（不含 pip 缓存与构建工具，体积最小化） ─────────────────
FROM python:3.12-slim

# 安全基线：非 root 运行
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 依赖（来自 builder 的 site-packages）
COPY --from=builder /deps /deps

# 应用代码（serve.py 位于 src/，--app-dir 保证平铺模块可导入）
COPY src ./src
COPY index.html ./

USER appuser

ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 轻量健康检查：/health 不触发索引预热，毫秒级返回
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "--app-dir", "src", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

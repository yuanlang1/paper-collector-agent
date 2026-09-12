# paper-collector-agent

## 在 Docker 中运行 Nacos

项目仅通过 Nacos 做服务注册与发现，当前 `.env` 的
`NACOS_SERVER_ADDR="127.0.0.1:8848"` 可保持不变。根目录的
`compose.nacos.yml` 使用与本机安装一致的 Nacos `3.2.0` 单机 Derby
模式；数据和日志存放在 Docker 命名卷中，重建容器不会丢失。

在 PowerShell 中启动：

```powershell
docker compose -f compose.nacos.yml up -d
```

随后可访问 [Nacos 控制台](http://127.0.0.1:8080/)，或使用以下命令检查容器：

```powershell
docker compose -f compose.nacos.yml ps
docker compose -f compose.nacos.yml logs -f nacos
```

该配置保持本机 Nacos 的 `NACOS_AUTH_ENABLE=false`。它只适用于本机开发；若将端口暴露到非受信任网络，应先启用认证，并替换 Compose 中的三个 `NACOS_AUTH_*` 值及限制端口访问范围。

停止服务时保留数据：

```powershell
docker compose -f compose.nacos.yml down
```

若要连同 Nacos 的 Docker 数据一并移除，执行 `docker compose -f compose.nacos.yml down -v`。

## 安装 Python 依赖

本项目以 Python 3.11 为基准。请使用锁定文件安装完整的直接与传递依赖：

```powershell
python -m pip install -r requirements.venv311.lock.txt
```

该文件包含 CUDA 12.8 版 PyTorch 的官方 wheel 源；没有 NVIDIA/CUDA 环境时，请按目标环境重新生成锁定文件。

## LLM 配置档案

前端可通过 `/api/settings/llm-profiles` 管理 OpenAI-compatible 与 DeepSeek 的多个 LLM 配置档案，并在调用 `/api/agent/chat` 或 `/api/agent/chat/stream` 时传入 `llm_profile_id`。未指定时使用默认档案，未创建默认档案时回退到现有 `.env` 配置。

启用配置档案前，必须在服务端 `.env` 设置用于加密 API Key 的 Fernet 密钥：

```powershell
.\.venv311\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

将输出写入 `LLM_PROFILE_ENCRYPTION_KEY`。请妥善保管该密钥；更换或丢失后，现有档案中的 API Key 将无法解密。

## 预下载 RAG 模型

在项目根目录执行以下 PowerShell 命令，将 RAG 模型保存到本地 `models/` 目录：

```powershell
# 稠密向量模型：运行时直接从 models/bge-m3 加载
.\.venv311\Scripts\hf.exe download BAAI/bge-m3 --local-dir models\bge-m3

# 稀疏向量模型：保留 Hugging Face 缓存结构，供 FastEmbed 从 models/fastembed 加载
.\.venv311\Scripts\hf.exe download Qdrant/bm25 --cache-dir models\fastembed
```

模型下载完成后可离线启动服务。`models/` 已被 Git 忽略，不会提交模型文件。

## 运行论文 RAG 评测

评测模块独立于主图和线上索引。PDF 论文重新索引后，使用以下命令生成带页码证据的候选集：

```powershell
python -m app.rag.evaluation generate --dataset paper-content-v1 --paper-id 120 --candidates-per-paper 5
python -m app.rag.evaluation review --dataset paper-content-v1 --case-id CASE_ID --status approved --reviewer reviewer
python -m app.rag.evaluation publish --dataset paper-content-v1
python -m app.rag.evaluation run --dataset paper-content-v1 --top-k 20
```

Ragas 指标使用独立 Python 3.11 虚拟环境，避免升级服务运行时的 LangChain 依赖：

```powershell
python -m venv .venv-rag-eval
.\.venv-rag-eval\Scripts\python.exe -m pip install -r requirements.venv311.lock.txt
.\.venv-rag-eval\Scripts\python.exe -m pip install -r requirements.rag-eval.venv311.txt
```

随后使用该环境执行 `python -m app.rag.evaluation run --metrics deterministic ragas`。

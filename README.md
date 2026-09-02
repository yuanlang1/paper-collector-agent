# paper-collector-agent

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

# paper-collector-agent

## 安装 Python 依赖

本项目以 Python 3.11 为基准。请使用锁定文件安装完整的直接与传递依赖：

```powershell
python -m pip install -r requirements.venv311.lock.txt
```

该文件包含 CUDA 12.8 版 PyTorch 的官方 wheel 源；没有 NVIDIA/CUDA 环境时，请按目标环境重新生成锁定文件。

## 预下载 RAG 模型

在项目根目录执行以下 PowerShell 命令，将 RAG 模型保存到本地 `models/` 目录：

```powershell
# 稠密向量模型：运行时直接从 models/bge-m3 加载
.\.venv311\Scripts\hf.exe download BAAI/bge-m3 --local-dir models\bge-m3

# 稀疏向量模型：保留 Hugging Face 缓存结构，供 FastEmbed 从 models/fastembed 加载
.\.venv311\Scripts\hf.exe download Qdrant/bm25 --cache-dir models\fastembed
```

模型下载完成后可离线启动服务。`models/` 已被 Git 忽略，不会提交模型文件。

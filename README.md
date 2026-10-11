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

## 内部 gRPC 接口

本服务只暴露根路径 `/` 和健康检查 `/health` 作为 HTTP 接口。主服务完成用户鉴权和权限判断后，通过内网 gRPC 调用本服务；业务 HTTP 接口、Swagger、ReDoc 和 OpenAPI 均已关闭。

默认监听 `0.0.0.0:50052`，可通过 `.env` 配置以下值：

```dotenv
AGENT_GRPC_HOST=0.0.0.0
AGENT_GRPC_PORT=50052
AGENT_GRPC_MAX_MESSAGE_LENGTH=16777216
```

服务以 `SERVICE_NAME` 注册一次；注册 metadata 中的 `grpc_port` 表示内部 gRPC 端口。主服务发现 `paper-collector-agent` 实例后，使用实例 IP 和该 metadata 端口建立 gRPC 连接。必须通过网络策略限制该端口只允许主服务访问。

协议位于 [`app/protos/agent/v1/agent.proto`](app/protos/agent/v1/agent.proto)，包括：

- `AgentService`：启动、恢复、查询和订阅运行事件；运行与事件请求均要求非空 `user_id`、`conversation_id`、`run_id`。
- `ConversationService`：会话与消息查询、会话删除。
- `SettingsService`：检索来源限制和 LLM 配置档案管理。

主服务可使用以下命令生成 Python 客户端：

```powershell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. app/protos/agent/v1/agent.proto
```

最小调用示例：

```python
import grpc
from app.protos.agent.v1 import agent_pb2, agent_pb2_grpc

async with grpc.aio.insecure_channel("agent-service:50052") as channel:
    agent = agent_pb2_grpc.AgentServiceStub(channel)
    started = await agent.StartRun(agent_pb2.StartRunRequest(
        user_id="verified-user-id",
        conversation_id="conversation-1",
        run_id="run-1",
        message="检索多模态大模型视觉 token 压缩论文",
    ))
    async for event in agent.SubscribeRunEvents(
        agent_pb2.SubscribeRunEventsRequest(
            user_id="verified-user-id",
            conversation_id=started.conversation_id,
            run_id=started.run_id,
        )
    ):
        print(event.sequence, event.event_name, event.data_json)
```

## LLM 配置档案

主服务通过 `SettingsService` 管理 OpenAI-compatible 与 DeepSeek 的多个 LLM 配置档案，并在 `AgentService.StartRun` 中传入可选 `llm_profile_id`。未指定时使用默认档案，未创建默认档案时回退到现有 `.env` 配置。

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

## 综述生成流程

综述采用固定任务论文集：复用逐篇阅读结果 → 明确研究问题 → 候选综合 → 原文检索与核验 → 大纲与写作 → 审核与保存。systematic/scoping 仅影响组织方式，不代表执行正式系统综述协议。

- 新增配置 `PAPER_READING_COLLECTION_NAME`，默认 `paper_reading`，首次保存时按需创建。复用原有 Qdrant dense/sparse 索引，不迁移或删除正文索引。
- 阅读在 extract_studies 的普通循环中处理，每批最多三篇，每篇成功立即保存。共享记录只含论文 ID 与 core_problem、methods、main_discussion，不保存 chunks 或工具消息。
- 缓存按论文 ID 精确命中；命中时不读取正文，也不调用阅读模型。论文正文替换或需要按新规则重新阅读时，删除该论文的缓存记录。
- 缓存保存失败最多尝试三次，不重新调用阅读模型；最终失败则返回失败论文信息。
- 原文证据和核验摘录独立保存在任务 artifact。只有 supported 的当前 Claim 措辞可写入；mixed/contradicted/insufficient 进入修订。
- Claim 核验最多执行 max_reflection_rounds 次（默认 5 次，包含首次核验），因此最多进行 max_reflection_rounds - 1 次 Claim 修订；预算耗尽时保留已核验内容，并将未支持内容写为材料不足。全文写作修订使用独立的 max_writing_revision_rounds。失败时返回已有产物引用，不保存为审核通过的综述。
- 部署新版前排空运行中的旧综述任务。旧 artifact 文件仍可读取，旧中间 checkpoint 不支持恢复到新版流程。综述子图递归限制独立为 128。

离线回归测试（假模型、假存储）：

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_review*.py' -q
```

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

本项目使用 Python 3.11 和 uv。`pyproject.toml` 声明依赖，`uv.lock` 锁定
Windows/Linux 的解析结果。默认安装 `app` 依赖组，保留当前应用运行环境；
不安装该组时，仅安装后台 Worker 所需的依赖。

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv311"
uv sync --locked --inexact
uv pip check --python .\.venv311\Scripts\python.exe
```

`--inexact` 保留现有环境中额外安装的工具。`app` 组沿用 CUDA 12.8 版
PyTorch，uv 已为 `torch` 和 `torchvision` 配置独立官方源；Windows 专有包
包含平台标记。依赖配置统一维护在 `pyproject.toml` 与 `uv.lock`，
后续依赖更新使用 `uv add`，并提交这两个文件。

## PostgreSQL 与 Redis

记忆存储复用 PostgreSQL 17 的 `agent_db` 和现有数据卷
`paper_agent_pgdata`。`Dockerfile.postgres` 基于原镜像安装 pgvector 0.8.7、
SCWS 和固定提交的 zhparser；Compose 将该数据卷声明为 external，不新建空库。
首次接管独立启动的 `paper-agent-postgres` 容器时，须先备份、停止并重命名
原容器，再由 Compose 启动；同一数据卷不可同时运行两个 PostgreSQL 实例。

在本机 `.env` 配置以下变量，数据库凭据沿用现有配置，宿主机地址为
`127.0.0.1:5432`：

```dotenv
DATABASE_POSTGRES_URL='postgresql://yl:URL编码后的原密码@127.0.0.1:5432/agent_db'
SILICONFLOW_API_KEY='你的密钥'
SILICONFLOW_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
DENSE_EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

在 Docker Desktop 的 Linux 容器模式下启动 PostgreSQL 和 Redis：

```powershell
docker compose -f compose.memory.yml up -d --build
docker compose -f compose.memory.yml exec redis redis-cli ping
```

PostgreSQL 已运行时，可以单独启动 Redis：

```powershell
docker compose -f compose.memory.yml up -d redis
```

Redis 使用 `127.0.0.1:6380`，与已有的 6379 实例分开；AOF 文件保存在
`memory-redis-data` 命名卷中。本地 Celery 默认连接 `redis://127.0.0.1:6380/0`。
可用环境变量 `CELERY_BROKER_URL` 和 `CELERY_RESULT_BACKEND` 覆盖地址。
任务结果保存在 Redis 的 DB 1，保留一小时。

FastAPI、Celery Worker 和 Beat 分别作为本地独立进程运行，Compose 仅管理
PostgreSQL 和 Redis。每个 completed run 在同一事务内创建 UserProfile、Fact
两个 job，提交后分别派发。Beat 每分钟检查空闲超过 10 分钟、最后一个 run 为
completed 的 thread，创建 Episode job，并补派发 pending job、恢复过期租约。

确认 PostgreSQL、Nacos 已运行，并完成下方记忆存储初始化后，在三个独立
PowerShell 终端中按 FastAPI、Worker、Beat 的顺序启动。FastAPI 启动时创建缺失的
业务表，包括 `memory_job`；等待日志显示 `Application startup complete` 后，再启动
Worker 和 Beat。每个终端先执行：

```powershell
cd E:\FastAPI_project\paper-collector-agent
$env:UV_PROJECT_ENVIRONMENT = ".venv311"
```

终端 1：启动 FastAPI；Windows 使用 Selector 事件循环以支持 Psycopg 异步连接：

```powershell
uv run --locked --no-sync --env-file .env python -m uvicorn app.main:app --loop asyncio:SelectorEventLoop --host 127.0.0.1 --port 8000
```

终端 2：启动串行 Worker，执行 UserProfile、Fact 和 Episode 提取任务：

```powershell
uv run --locked --no-sync --env-file .env celery -A app.tasks.celery_app:celery_app worker --pool=solo --concurrency=1 --queues=memory --loglevel=INFO
```

终端 3：启动 Beat，定期扫描 Episode、补派发 pending job 并恢复过期租约：

```powershell
uv run --locked --no-sync --env-file .env celery -A app.tasks.celery_app:celery_app beat --schedule data/celerybeat-schedule --loglevel=INFO
```

提取使用对应用户配置中 `is_small_model=true` 的小模型。当前只运行一个 `solo`
Worker 和一个 Beat，避免同一用户的
画像被并发覆盖；增加 Worker 前需要用户级串行控制。租约为 5 分钟，每分钟续租，
失败最多尝试 3 次。任务只传递 job_id，根据固定 checkpoint 与 end_id 读取消息。
提取采用至少一次执行；Store 写入与 job 完成不属于同一个事务，崩溃重试存在重复
写入窗口。主图前的记忆检索与提示词注入尚未接入。

Worker 通过 `worker_process_init` 创建进程专属事件循环、AsyncEngine 和
`async_sessionmaker`，数据库使用 `NullPool`。三个提取任务和维护任务统一通过
`run_async_task` 在该循环执行，每个任务使用独立 AsyncSession；领取、续租和完成
分别提交短事务，续租协程使用另一个 Session。正常退出时清理遗留协程、数据库连接
和事件循环，`solo` 模式同时通过 `worker_shutdown` 完成清理。

任务模块统一列在 `app/tasks/celery_app.py` 的 `TASK_MODULES` 中，由 Worker
启动时自动导入。`app/tasks/modules/memory/` 下的 `extract_userprofile.py`、
`extract_fact.py`、`extract_episode.py`、`maintain.py`、`healthcheck.py` 各注册
一个任务，`common.py` 保存公共辅助函数。新增任务文件时同步加入模块清单。

首次启动记忆模块前，从项目根目录在本地初始化记忆存储：

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv311"
uv run --locked --no-sync --env-file .env python -m app.memory.init_store
```

记忆数据库初始化命令可重复执行：由原生 LangGraph Store 创建 `halfvec(1024)`
表和 cosine HNSW 索引，并检查实际列类型和索引。该命令负责 Store 相关表，
`memory_job` 等业务表由 FastAPI 启动时创建。需要同时验证 embedding 时，在命令末尾
添加 `--check-embedding`，会调用一次 SiliconFlow；不带此参数时只初始化数据库。
现有其他维度或不同类型的索引须单独迁移。
PostgreSQL 设置 `hnsw.ef_search=100`、`hnsw.iterative_scan=strict_order`；
当前使用 Store 原生语义检索，初始化命令不会启动记忆提取。
原有论文 RAG 的 Qdrant 配置保留。

切换前备份保存在 Git 忽略的 `data/backups/postgres/`；不要删除原数据卷。

查看日志或停止服务并保留数据：

```powershell
docker compose -f compose.memory.yml logs --tail 100 postgres redis
docker compose -f compose.memory.yml down
```

## 内部 HTTP 接口

HTTP 提供根路径 `/`、健康检查 `/health`、Agent SSE 与历史接口、thread 操作和 run 查询。主服务负责用户鉴权和权限判断。内部检索数量配置使用 `GET /internal/settings/source-limits` 和 `PUT /internal/settings/source-limits`；更新请求为 `{"arxiv": 20, "dblp": 20, "google_scholar": 20}`，返回 `ServiceResponse`，其中 `data` 包含 `limits`、`page_sizes` 和 `maximum_limits`。Swagger、ReDoc 和 OpenAPI 均已关闭。

Agent 仅监听原 HTTP 端口，并以 `SERVICE_NAME` 在 Nacos 注册。主服务通过实例 IP 和 HTTP 端口调用内部配置接口；网关不得对外转发 `/internal/**`，需通过网络策略限制内部接口只允许可信服务访问。旧配置调用方需同步切换到 HTTP，不再提供业务 gRPC 服务。

访问 Java paper-service 时复用 `httpx.AsyncClient`，调用 `/internal/**` 接口。`.env` 配置如下：

```dotenv
PAPER_SERVICE_NAME=paper-service
PAPER_SERVICE_BASE_URL=
PAPER_SERVICE_CONNECT_TIMEOUT_SECONDS=5.0
PAPER_SERVICE_TIMEOUT_SECONDS=10.0
```

`PAPER_SERVICE_BASE_URL` 留空时，从 Nacos 发现 `paper-service` 的 HTTP 实例；也可指定实际地址（如 `http://127.0.0.1:8080`）直连。客户端同时检查 HTTP 状态和响应体的 `success`、`code`，写操作不自动重试。Nacos 与 Qdrant SDK 及其底层依赖继续保留。

协议位于 [app/protos/agent/v1/agent.proto](app/protos/agent/v1/agent.proto)。当前只注册 `SettingsService.GetSourceLimits` 和 `UpdateSourceLimits`。旧主 Agent 图、会话历史和 LLM 配置档案的运行入口已停用；保留协议定义，相关 RPC 返回 `UNIMPLEMENTED`。论文检索、RAG 索引和综述生成的业务工作流仍保留，等待新主架构接入。

## 数据库生命周期

数据库引擎和异步会话工厂由 FastAPI `lifespan` 调用 `create_database()` 创建，关闭时释放连接池。会话工厂使用 `async_sessionmaker(bind=engine, expire_on_commit=False)`。HTTP 从 `app.state.session_factory` 取得工厂；每次操作创建独立 `AsyncSession`。Celery Worker 创建自己的 AsyncEngine 和会话工厂，与 Web 服务分开管理。

`app/database.py` 统一定义 ORM 基类、数据库引擎工厂和异步会话依赖；`app/core/database.py`、`app/db/session.py` 复用同一工厂。`app/db/models/` 按表定义模型，`app/db/repositories/` 使用 `AsyncSession` 执行原生 SQL。业务服务负责校验和事务，仓储不自行 commit。

业务数据库统一使用 `DATABASE_POSTGRES_URL=postgresql://user:password@host:5432/database`。SQLAlchemy 使用 `postgresql+psycopg` 异步驱动，LangGraph checkpointer 和 store 使用原始连接地址。启动时通过 `Base.metadata.create_all()` 创建缺失业务表，不升级已有表结构，也不搬迁旧数据库数据。

Windows 上 Psycopg 异步连接需要 Selector 事件循环。使用当前锁定的 Uvicorn 启动时指定：`python -m uvicorn app.main:app --loop asyncio:SelectorEventLoop`；自行使用 `asyncio.run()` 时，需在创建循环前设置 `asyncio.WindowsSelectorEventLoopPolicy()`。

七张业务表统一继承 `EntityBase`，共用自增 ID、创建人、修改人、创建时间和修改时间。时间字段使用带时区类型；SQLAlchemy 更新时维护 `modification_time`，原生 SQL 和 Upsert 需显式赋值。`llm_setting` 保存模型配置，主图按请求的 `metadata.model_id` 和 `user_id` 读取；SQLite 事件回放存储继续独立使用。

`llm_setting.user_id` 表示配置所属用户，与审计操作人独立；配置 CRUD 和 `get_llm_config()` 均要求显式传入 `user_id`。已有空表可执行 [新增用户字段 SQL](migrations/20261020_add_llm_setting_user_id.sql)。有历史数据的表需要单独新增该列、按实际归属回填历史记录，再设置非空约束和索引；启动建表不会升级已有表，也不会根据创建人自动推断归属。

`agent_thread` 和 `agent_run` 显式保存并按 `user_id` 过滤；`config_setting` 继续保持系统级配置。run 执行由 `agent_run_service` 管理，历史查询由 `thread_service` 校验会话归属后读取 LangGraph checkpoint，消息及图状态保存在 checkpoint 中。

`agent_run` 为每次被接受的 Agent 请求保存独立 `run_id`、模型参数、状态及起止时间。`POST /api/agent/run` 接收正整数 `user_id`、可选的字符串 `thread_id`、必填的 `metadata`，以及二选一的 `message` 或 `resume`。`metadata` 包含正整数 `model_id` 和可选的 `think_effort`，直接保存到 `agent_run.metadata` JSON 列；消息和恢复值仅通过请求传给 Runner，不存入 run 表。ORM 使用 `run_metadata` 属性映射该列，避免覆盖 SQLAlchemy 的 `metadata`；run 查询对外返回平铺的 `metadata` 字段。恢复请求同样显式传递 `metadata` 并创建新 run，原 run 保留 `interrupted`。状态包括 `pending`、`running`、`completed`、`failed`、`interrupted`、`cancelled`。创建及状态变化与 thread 更新时间在同一短事务内提交；同一 thread 只允许一个 `pending/running` run，并发执行或删除活跃 thread 返回 `409`。

发送 `message` 时，`thread_id` 未传、为 `null` 或空字符串均表示新会话，服务端生成 ID 并在同一事务内创建 thread 和首个 run，`topic` 为 `null`；run 入库失败时两者一起回滚，执行阶段失败则保留 thread 和 `failed` run。非空 ID 校验会话归属，不存在或属于其他用户时返回 `404`；`resume` 必须提供非空 `thread_id`。SSE 首条状态事件及后续事件携带实际的 `thread_id` 和 `run_id`，前端保存生成的会话 ID 用于后续请求。

已有数据库需要停写后执行 [字段迁移 SQL](migrations/20261020_rename_agent_run_input_to_metadata.sql)，再启动新版代码；`create_all()` 不会自动改列。脚本针对当前 `input.metadata` 的 JSON 格式，将列改名并仅保留其中的模型参数，去掉旧 JSON 中的消息和恢复值；整个迁移在同一事务内执行。

针对本次检查到的空业务表旧结构，可使用 [统一迁移 SQL](migrations/20261020_align_business_tables.sql)：统一字段类型、时间字段和约束，并删除 `agent_session`、`agent_message`、`agent_summary`。它包含上述两份独立迁移的改动，不应重复执行；脚本在同一事务内锁定目标表，发现任何业务记录时中止，LangGraph 自有表不参与迁移。

主图在创建 `AgentRunner` 前调用 `agent_build.get_model()` 和 `build_tools()`，分别注入模型和工具注册表。模型使用所选配置的 `model`、`base_url`、`api_key` 直接构建 `ChatOpenAI(streaming=True)`；`metadata.think_effort` 映射到 `reasoning_effort`，合法取值复用当前 SDK 的 `ReasoningEffort` 类型，不传或传 `null` 时不向模型接口发送该参数。具体模型不支持该参数时沿用 run 的失败处理。`provider` 字段保留，暂不参与主图模型构建。子代理内部的模型构建继续使用各自现有逻辑。

`GET /api/agent/threads/{thread_id}/runs` 支持 `user_id`、`limit`、`offset`，按 ID 倒序分页；`GET /api/agent/threads/{thread_id}/runs/{run_id}` 查询单次请求。SSE 事件包含 `run_id` 和 `thread_id`，终止事件在最终状态提交后发送。图中断时先发送 `type: "interrupt"`，包含 `status: "interrupted"` 和完整的 `interrupts: [{id, value}]`，再发送保留相同中断数据的 `done` 事件；`done` 表示本次流结束，执行结果以 `status` 为准。前端按 SSE JSON `data` 中的 `type` 分发事件。断连取消会关闭图流并将尚未结束的 run 记录为 `cancelled`；已提交的 `interrupted` 等终态保持不变。进程被强制终止后的遗留状态恢复尚未实现。

数据库集成测试只读取显式设置的独立测试库地址，在随机隔离 schema 内运行并清理；没有该变量时跳过数据库测试：

```powershell
$env:TEST_DATABASE_POSTGRES_URL = "postgresql://user:password@localhost:5432/paper_collector_test"
.\.venv311\Scripts\python.exe -m unittest tests.test_async_database tests.test_postgres_models tests.test_agent_runs -v
```

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
$env:UV_PROJECT_ENVIRONMENT = ".venv-rag-eval"
uv sync --locked --inexact
uv pip install --python .\.venv-rag-eval\Scripts\python.exe ragas
$env:UV_PROJECT_ENVIRONMENT = ".venv311"
```

评测附加依赖只安装到独立环境，不修改主服务的锁定文件。随后执行：

```powershell
.\.venv-rag-eval\Scripts\python.exe -m app.rag.evaluation run --metrics deterministic ragas
```

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

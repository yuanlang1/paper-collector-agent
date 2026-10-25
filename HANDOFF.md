# 会话交接：原生 SQL 改造与长期记忆接入准备

更新日期：2026-10-23。项目：`E:\FastAPI_project\paper-collector-agent`，Windows / PowerShell，Python 3.11，依赖使用 uv 管理。

## 1. 给新会话的第一条说明

本次会话的代码修改已经完成，最新请求只是编写本交接文件。当前没有正在执行的测试、待解决的审批或必须立即继续的代码任务。不要仅凭本文件自行扩大修改范围；先结合用户的新请求处理。

工作区有大量**尚未提交**的改动，包含本次原生 SQL 改造，也包含此前已有的小模型支持。不要 reset、checkout 覆盖、清理工作区或把所有改动当作自己新写的内容。

用户此前明确要求**不要查看 AOCI**。本次没有读取 `aoci.txt`，没有调用 AOCI，也没有维护其索引。不要因仓库 AGENTS.md 中的 AOCI 集成块而自动启动索引读取或维护。

用户多次明确要求使用 `karpathy-ponytail`，其文件为 `C:\Users\23922\.codex\skills\karpathy-ponytail\SKILL.md`。遵循最小修改、先追踪调用方、定位根因、使用现有依赖、完成验证的原则。没有必要重新设计架构。

## 2. 我们在做什么

项目是 FastAPI + LangGraph 的论文助手，有 PostgreSQL 业务表、LangGraph PostgreSQL checkpointer、长期记忆 Store，以及 Celery/Redis 基础配置。

最近一阶段用户要求：

1. 将 `app/db/repositories` 的业务运行时 CRUD 改为 SQLAlchemy `text()` 参数化原生 SQL，并同步适配调用方。
2. 在保证权限、事务和并发正确的前提下，适合连表查询的位置使用 JOIN，减少多余往返。
3. 检查改造结果，再修复发现的问题。

最后一次代码请求：`message` 为 `None` 时不要切片，并修复另外两个问题：配置时间戳测试仍按 ORM 对象访问、ASGI 2.3 断连时 LangGraph 清理不完整。

## 3. 已完成：原生 SQL 改造

### 仓储与返回值

以下仓储已改为参数化原生 SQL：

- `app/db/repositories/agent_run.py`
- `app/db/repositories/agent_thread.py`
- `app/db/repositories/llm_setting.py`
- `app/db/repositories/system_setting.py`
- `app/db/repositories/memory_job.py`

业务多列查询返回 `RowMapping`，调用方使用 `row["field"]`，不再按 ORM 实例访问。标量接口保持标量，例如 `system_setting.get()` 返回 `str | None`。

`app/db/models` 仍保留 SQLAlchemy ORM 声明，用于表结构、DDL、约束和索引；这不意味着业务查询仍走 ORM。本轮没有删除模型，也没有修改业务表结构。

原生 SQL 中，数据库列 `agent_run.metadata` 显式别名为 `run_metadata`，兼容业务调用方。JSON 参数使用明确的 JSON 类型绑定，不要把 Python 字典直接交给未声明类型的 SQL 参数。

### 已同步的调用方

- `app/services/agent_run_service.py`
- `app/services/thread_service.py`
- `app/services/llm_setting_service.py`
- `app/services/setting_service.py`
- `app/services/internal_settings_service.py`
- `app/api/run.py`
- `app/api/setting.py`

`llm_setting` 已有按用户获取小模型的方法；服务层入口为 `get_small_llm_config(db, *, user_id)`。该能力最初部分代码就是既有工作区改动，已保留并适配，不能误删。

### 必须保留的查询与并发语义

- run 列表用 `LEFT JOIN LATERAL` 同时确认 thread 归属并分页。合法用户的空会话、越界页返回 `[]`；不存在或不属于用户的会话返回 404。不能用普通 INNER JOIN 把这些情况混在一起。
- 在已有 thread 上创建 run 时，`agent_thread.touch()` 通过 `UPDATE ... RETURNING` 验证归属、加锁并更新时间，在同一事务创建 run。
- run 状态更新仍先锁 thread，再通过期望状态执行 CAS 更新；只有 run 真正发生状态迁移才 touch thread。不得让断连的 cancelled 覆盖已提交的 completed/interrupted。
- 删除 thread 仍先锁 thread，再用**新的一条语句**检查 active run。不要为减少 SQL 数量合成一个语句：等待锁后需要 READ COMMITTED 的新快照，否则会漏掉并发提交的 run。
- 删除 checkpoint 的 SQL 已连表验证 thread 所属用户，不能删除其他用户的 checkpoint。
- `memory_job` 创建通过 source CTE JOIN 验证 user/thread/completed run。每轮 job 保留“插入、再查询”的两条语句，以看到并发插入者提交的记录；不能盲目合并成一个 CTE 后读取旧语句快照。
- job 的 claim token、租约、`SKIP LOCKED`、幂等约束及过期恢复语义已保留。
- 原生 SQL 不会执行 ORM 的 Python `onupdate`。需要更新时间的 UPDATE 必须显式写 `modification_time`。LLM 配置和 topic 的无变化更新保留原时间戳。

## 4. 已完成：最后三个修复

### `app/services/thread_service.py`

创建会话使用：

```python
topic = message[:40] if message is not None else None
```

保存到仓储和返回 DTO 都使用计算后的 `topic`。因此：None 保持 None，空字符串保持空字符串，长内容截取前 40 字；数据库值与 API 返回值一致。

`tests/test_agent_runs.py::AgentRunTests.test_create_thread_with_optional_message` 验证省略 topic、显式 null、空字符串和长字符串，同时检查 API 返回值及数据库保存值。

### `tests/test_async_database.py`

`test_postgres_defaults_and_automatic_update_time` 原先把 `system_setting.get()` 当 ORM 对象并修改属性，原生 SQL 改造后会报 `'str' object has no attribute 'modification_time'`。

现在测试用参数化 UPDATE 设置旧时间、调用仓储更新配置、用参数化 SELECT 查询时间列，并独立断言 getter 返回字符串 `second`。没有为了旧测试重新改变生产接口。

### `app/api/run.py`

ASGI 2.3 的 Starlette 断连处理使用 AnyIO 取消作用域，其重复取消会打断 LangGraph 退出时的异步清理。仅在响应最外层 finally 加 `CancelScope(shield=True)` 已经太晚，底层清理可能早已中断。

现新增 `RunStreamingResponse.stream_response()`：

- 用独立 `asyncio.create_task(super().stream_response(send))` 运行流响应。
- 正常等待使用 `asyncio.shield(task)`。
- 收到取消后显式取消该 task，在 AnyIO shield 中等待其退出，然后继续抛出取消。
- 原有 `__call__` 的关闭生成器与取消 run 逻辑保留，依靠 CAS 保留已完成状态。

没有改 LangGraph 第三方源码，也没有扫描或取消所有 event-loop task。

已有断连回归测试覆盖 ASGI 2.3 和 2.4。测试的图节点 finally 中还包含一次异步让出，确认不只是进入 finally，而是可以完成异步清理。发送失败、已中断、已完成状态保持的测试也通过。

## 5. 测试结果与明确的遗留问题

### 最新验证

1. 五个定向测试全部通过：可选 topic、运行中断连、发送失败、interrupt 后断连、配置默认值/更新时间。
2. 以下五个测试模块合计 **62 项通过**，运行时间约 32 秒：
   - `tests.test_postgres_models`
   - `tests.test_memory_jobs`
   - `tests.test_small_llm_config`
   - `tests.test_agent_runs`
   - `tests.test_async_database`
3. 上述回归**主动排除了下面 7 项此前已存在的旧测试兼容问题**，所以不能说全套测试已通过。
4. `git diff --check` 通过；存在 Git 的 CRLF/LF 提示，无需为此格式化整个项目。
5. 集成测试使用随机创建的独立 PostgreSQL 临时数据库，结束后删除；没有使用业务数据执行清理测试。

### 未处理的 7 项旧测试

全部位于 `tests/test_async_database.py`：

```text
test_llm_crud_filter_pagination_and_rollback
test_llm_service_commits_crud_and_filters_settings
test_llm_service_returns_config_and_reports_missing_setting
test_llm_service_rolls_back_failed_update
test_llm_settings_are_scoped_to_owner_not_operator
test_lifespan_releases_database_even_when_startup_fails
test_lifespan_stops_services_and_disposes_database_on_shutdown
```

前五项仍传已删除的 `operator_id`，而且部分断言仍按旧 ORM 对象/API 行为编写。当前生产 `llm_setting.create/update` 没有该参数。不要为满足过时测试，重新添加用户未要求的 operator 接口；应按当前真实接口更新测试，并保留用户隔离、分页、提交与回滚验证。

后两项 monkeypatch 不存在的 `app.main.AsyncPostgresStore`。当前 lifespan 已使用 `app.main.open_memory_store()`，需要替换为异步上下文管理器测试替身，再检查启动失败和正常退出时资源释放顺序。

最后修复的 ASGI 问题已用原 HEAD 模块做过基线复现，确认不是原生 SQL 改造引入；但现在已修复。不要再次把它当作未解决项。

## 6. 长期记忆的背景、当前实现与未接通部分

这是本项目更长期的需求，**不是下一步必须擅自实施的任务**。用户的新请求若回到 memory，再按当前代码确认细节。

### 已确定的方向

- 使用 LangMem `create_memory_store_manager` 维护 UserProfile、Fact、Episode。
- 每个 `completed` run 都提取 UserProfile 和 Fact，不再用小模型先做“是否提取”的路由判断，也不再每 4 轮触发。
- Episode 在 thread 一段时间无变化后提取，讨论值约 10 分钟，依据 thread 更新时间；不是每轮都提取 Episode。
- job 不保存消息正文快照，不记录 start_id、不保存 model_id；已有 checkpoint_id + end_id 用于定位读取边界。
- 提取使用用户自己的小模型配置，已有 `get_small_llm_config` 可复用。
- Celery Worker 在本地进程运行，与 FastAPI 是两个独立进程；不要重新添加 Docker Worker 部署。
- 长期记忆 Store 使用 PostgreSQL + pgvector，`halfvec(1024)` + HNSW + cosine；embedding 为 SiliconFlow `BAAI/bge-m3`，使用项目 `SiliconFlowEmbeddings` 的 dense 输出。
- 当前先使用 Store 原生语义检索。RRF、zhparser、手工拆词/OR 拼接等不要重新引入；混合检索是延期方向。
- Episode 按用户级检索：检索 prefix 是 `episode_user_prefix(user_id)`，不是限定当前 thread。写入 namespace 仍可包含 thread_id/job_id。
- 用户希望进入主图之前检索并拼入 system_prompt，不希望新增图内记忆检索节点。

### 当前已有的代码

- `app/memory/store.py`：`open_memory_store()`、1024 维 embedding、halfvec HNSW 配置，连接指定 public schema；HNSW 查询参数也已设置。
- `app/memory/init_store.py`：Store 初始化/结构检查入口；不要在未核实数据库结构时声称初始化已经保证一切正确。
- `app/memory/manager.py`：Profile/Fact/Episode manager 工厂已经存在。
- `app/memory/schemas.py`、`namespace.py`、`context.py`：记忆类型、namespace 与上下文构建已经存在；`context.py` 已按用户级查 Episode。
- `app/main.py`：lifespan 打开 Store，保存在 `app.state.store`，请求再传入 runner/graph。
- `app/db/models/memory_job.py` 与仓储：job 表及幂等、领取、续租、完成、重试、失败、跳过、过期恢复能力已经存在。
- `app/tasks/celery_app.py`：Celery 基础配置和 healthcheck；broker 默认 `redis://127.0.0.1:6380/0`，backend 默认 `/1`，队列 `memory`。实际运行环境与 Redis 状态需要另行核实。healthcheck 中仍出现 qdrant_client 版本字段，不要据此误认长期记忆还存 Qdrant，也不要顺手清理未经授权的其他 RAG 依赖。

### 当前未接通

截至交接，`app/services/agent_run_service.py::run()` 在完成状态提交后只发送 interrupt/done 事件，**没有调用 memory_job 创建与派发逻辑**。`app/tasks/celery_app.py` 目前只有 healthcheck，没有完整的三类记忆提取任务。静默 Episode 扫描、执行、恢复调度和主图前 system_prompt 注入仍需按后续请求实施。

不要把 manager 工厂、Store 注入、job 仓储已经存在，误报为后台提取全链路已经完成。

## 7. 复现测试的可靠方法

### 环境注意

本机同时存在 `.venv` 与 `.venv311`；之前 uv 提示当前激活的 `.venv311` 与默认项目 `.venv` 不匹配。当前验证使用 `.venv311`，且默认 uv cache 路径可能权限不足：

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv311'
uv --no-cache run --locked --no-sync python ...
```

这是已安装依赖环境下的测试方式，不要为了测试重新解析/升级依赖或改 lockfile。

Git 若提示 dubious ownership，使用单次命令配置：

```powershell
git -c safe.directory=E:/FastAPI_project/paper-collector-agent status --short
```

不要修改全局 Git 配置。

### 独立临时 PostgreSQL 测试库示例

将下面代码通过 PowerShell 单引号 here-string 管道传给上述 uv 的 `python -`。不得打印数据库连接串或秘钥。测试库名必须由本次随机生成，删除操作只能针对该库。

```python
import os
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url
from app.config import settings

name = "test_handoff_" + uuid4().hex
with psycopg.connect(
    settings.DATABASE_POSTGRES_URL, autocommit=True, connect_timeout=5
) as connection:
    connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        os.environ["TEST_DATABASE_POSTGRES_URL"] = (
            make_url(settings.DATABASE_POSTGRES_URL)
            .set(database=name).render_as_string(hide_password=False)
        )
        # 必须先设置环境变量，再导入/加载测试模块。
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "tests.test_agent_runs",
            "tests.test_memory_jobs",
            "tests.test_small_llm_config",
            "tests.test_postgres_models",
            "tests.test_async_database.AsyncDatabaseTests.test_postgres_defaults_and_automatic_update_time",
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        connection.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
        )
raise SystemExit(not result.wasSuccessful())
```

这份示例选择相关模块及已修复单项，不等同于前述 62 项完整回归筛选。需要复跑完全相同的 62 项时，加载上述五个完整模块后，仅过滤第 5 节列出的 7 个方法；不要把过滤结果说成全套通过。

### SQL 测试隔离的关键点

`PostgresDatabaseTestCase` 每个用例生成随机 schema，同时配置连接 `search_path` 和 `schema_translate_map`。**只有 schema_translate_map 不够：SQLAlchemy `text()` 完全不会按该映射改写表名。** 删掉 search_path 会让原生 SQL 访问 public，破坏隔离。

`AgentRunTests` 的 saver 与业务 SQL 连接也必须使用同一测试 schema。`TEST_DATABASE_POSTGRES_URL` 在测试模块导入时读取，必须在 unittest 加载前设置。

## 8. 当前工作区文件清单

交接前 `git status --short` 显示以下已跟踪文件修改：

```text
.gitignore
app/api/run.py
app/api/setting.py
app/db/repositories/agent_run.py
app/db/repositories/agent_thread.py
app/db/repositories/llm_setting.py
app/db/repositories/memory_job.py
app/db/repositories/system_setting.py
app/services/agent_run_service.py
app/services/internal_settings_service.py
app/services/llm_setting_service.py
app/services/setting_service.py
app/services/thread_service.py
tests/test_agent_runs.py
tests/test_async_database.py
tests/test_memory_jobs.py
tests/test_postgres_models.py
```

`tests/test_small_llm_config.py` 是原先被忽略的既有测试文件，已更新并通过 `.gitignore` 例外允许跟踪，目前仍是 untracked。若用户后续要求提交，别漏掉它。本 `HANDOFF.md` 也是新文件。没有创建提交、PR 或推送。

## 9. 下一步建议与绝对不要再踩的坑

目前没有阻塞条件。若用户要继续收尾，优先更新那 7 项旧测试，然后在独立库跑完整相关模块；若用户要继续 memory，则先实现完成 run 与 job 的事务衔接/派发，再处理 worker、静默 Episode 和主图前上下文注入，分别验证，避免一次改完所有层。

- 不要回退或覆盖现有未提交代码，也不要重新从 ORM 方案开始。
- 不要用修改生产接口来迎合过时测试，不要重新添加 `operator_id`。
- 不要把 RowMapping 当 ORM 实例；也不要把所有返回值都当 RowMapping，配置 getter 是字符串。
- 不要误把 ORM `onupdate` 当 PostgreSQL 触发器。
- 不要为减少一次查询而破坏锁等待后的新快照语义，或破坏幂等插入后的并发可见性。
- 不要弱化 ASGI 2.3 断连测试、删除异步 finally 检查、用 sleep 掩盖泄漏，或修改第三方 LangGraph 源码。
- 不要直接用业务库做删除、清理、初始化迁移试验；不要输出 `.env` 或连接串。
- 不要认为 `init_store` 会把已有 `vector(1024)` 自动变成 `halfvec(1024)`：之前确实出现过类型不符报错。先只读确认表列与索引；用户当时说还没数据，不代表现在仍可直接删表。
- 不要恢复已放弃的每 4 轮触发、先分类再提取、start_id、job.model_id、Docker Worker 或当前 RRF/zhparser 方案。
- 不要自动删除 Qdrant 依赖，其他 RAG 功能仍可能使用。
- 不要宣称全套测试通过：目前明确结果是 62 项通过、7 项旧测试未纳入。

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError
from app.config import settings
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.grpc.server import InternalGrpcServer
from app.infrastructure.nacos_registry import nacos_registry
from app.core.exceptions import register_exception_handlers
from app.rag.index_construction.base import close_index_resources
from app.rag.index_construction.memory_index_sync import (
    get_memory_index_synchronizer,
)
from app.rag.processing.task_rag_batch_runner import close_task_rag_batch_runner
from app.runtime.agent_runtime import get_agent_runtime, initialize_agent_runtime
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.history.sqlite import initialize_history_database
from app.history.chat_log import AgentHistoryStore
from app.database import engine, SessionLocal
from app.models.llm_profile import LlmProfile
from app.models.memory import MemoryConsolidationCursor, MemoryEpisode, MemoryFact
from app.models.system_setting import SystemSetting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)


def _ensure_llm_profile_schema() -> None:
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("llm_profiles")
    }
    if "is_small_model" in columns:
        return

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE llm_profiles "
                    "ADD COLUMN is_small_model BOOLEAN NOT NULL DEFAULT FALSE"
                )
            )
    except OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise

# @app.on_event("startup")
# async def on_startup():
#     init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    SystemSetting.__table__.create(bind=engine, checkfirst=True)
    LlmProfile.__table__.create(bind=engine, checkfirst=True)
    _ensure_llm_profile_schema()
    MemoryFact.__table__.create(bind=engine, checkfirst=True)
    MemoryEpisode.__table__.create(bind=engine, checkfirst=True)
    MemoryConsolidationCursor.__table__.create(bind=engine, checkfirst=True)
    await get_memory_index_synchronizer().ensure_collections()
    async with AsyncSqliteSaver.from_conn_string(
        settings.LANGGRAPH_CHECKPOINT_PATH,
    ) as checkpointer:
        await checkpointer.setup()

        history_db = initialize_history_database(
            settings.CHAT_HISTORY_DB_PATH,
        )

        initialize_agent_runtime(
            checkpointer=checkpointer,
            history_db=history_db,
            history_store=AgentHistoryStore(SessionLocal),
        )

        grpc_server = InternalGrpcServer(get_agent_runtime())
        await grpc_server.start()
        try:
            await nacos_registry.start()
        except Exception:
            await grpc_server.stop()
            raise
        try:
            yield
        finally:
            await grpc_server.stop()
            await nacos_registry.stop()
            await get_agent_runtime().shutdown()
            await close_task_rag_batch_runner()
            await paper_service_grpc_channel_pool.close()
            await close_index_resources()

app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

register_exception_handlers(app)

# 根路由
@app.get("/")
async def root():
    """系统首页"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }

@app.get("/health")
async def health():
    return {
        "status": "UP",
        "service": settings.SERVICE_NAME,
    }

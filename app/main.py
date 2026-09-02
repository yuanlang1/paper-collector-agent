from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.config import settings
from app.api import *
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.infrastructure.nacos_registry import nacos_registry
from app.core.exceptions import register_exception_handlers
from app.rag.index_construction.base import close_index_resources
from app.rag.processing.task_rag_batch_runner import close_task_rag_batch_runner
from app.runtime.agent_runtime import initialize_agent_runtime
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.history.store import initialize_history_store
from app.database import engine
from app.models.llm_profile import LlmProfile
from app.models.memory import MemoryConsolidationCursor, MemoryEpisode, MemoryFact
from app.models.system_setting import SystemSetting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)

# @app.on_event("startup")
# async def on_startup():
#     init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    SystemSetting.__table__.create(bind=engine, checkfirst=True)
    LlmProfile.__table__.create(bind=engine, checkfirst=True)
    MemoryFact.__table__.create(bind=engine, checkfirst=True)
    MemoryEpisode.__table__.create(bind=engine, checkfirst=True)
    MemoryConsolidationCursor.__table__.create(bind=engine, checkfirst=True)
    async with AsyncSqliteSaver.from_conn_string(
        settings.LANGGRAPH_CHECKPOINT_PATH,
    ) as checkpointer:
        await checkpointer.setup()

        history_store = initialize_history_store(
            settings.CHAT_HISTORY_DB_PATH,
        )

        initialize_agent_runtime(
            checkpointer=checkpointer,
            history_store=history_store,
        )

        await nacos_registry.start()
        try:
            yield
        finally:
            await close_task_rag_batch_runner()
            await paper_service_grpc_channel_pool.close()
            await nacos_registry.stop()
            await close_index_resources()

app = FastAPI(lifespan = lifespan)

register_exception_handlers(app)

app.include_router(agent_router)
app.include_router(settings_router)

# 根路由
@app.get("/")
async def root():
    """系统首页"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "UP",
        "service": settings.SERVICE_NAME,
    }

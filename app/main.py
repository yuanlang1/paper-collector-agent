import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api import (
    internal_rag_router,
    internal_settings_router,
    run_router,
    setting_router,
    thread_router,
)
from app.common.exceptions import register_exception_handlers
from app.config import settings
from app.database import Base, create_database
from app.infrastructure.nacos_registry import nacos_registry
from app.infrastructure.redis import redis_manager
from app.infrastructure.http import http_manager
from app.infrastructure.oss import oss_store
from app.memory.store import open_memory_store
from app.rag.index_construction.base import close_index_resources
from app.services.search_connection_executor import search_executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as resources:
        engine, session_factory = create_database()
        resources.push_async_callback(engine.dispose)
        app.state.session_factory = session_factory
        resources.push_async_callback(redis_manager.close)
        await redis_manager.start()
        resources.push_async_callback(http_manager.close)
        await http_manager.start()
        search_executor.configure(session_factory)
        resources.callback(search_executor.reset)

        checkpointer = await resources.enter_async_context(
            AsyncPostgresSaver.from_conn_string(
                settings.DATABASE_POSTGRES_URL,
            )
        )
        await checkpointer.setup()

        store = await resources.enter_async_context(open_memory_store())

        app.state.checkpointer = checkpointer
        app.state.store = store

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        resources.push_async_callback(oss_store.close)
        resources.push_async_callback(close_index_resources)
        resources.push_async_callback(nacos_registry.stop)
        await nacos_registry.start()
        yield


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

register_exception_handlers(app)
app.include_router(run_router)
app.include_router(thread_router)
app.include_router(setting_router)
app.include_router(internal_settings_router)
app.include_router(internal_rag_router)


@app.get("/")
async def root():
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

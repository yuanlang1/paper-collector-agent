import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api import internal_settings_router, run_router, setting_router, thread_router
from app.common.exceptions import register_exception_handlers
from app.config import settings
from app.database import Base, create_database
from app.infrastructure.paper_service.http_client import paper_service_http_client
from app.infrastructure.nacos_registry import nacos_registry
from app.memory.store import open_memory_store
from app.rag.index_construction.base import close_index_resources
from app.rag.processing.task_rag_batch_runner import close_task_rag_batch_runner

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

        resources.push_async_callback(close_index_resources)
        await paper_service_http_client.start()
        resources.push_async_callback(paper_service_http_client.close)
        resources.push_async_callback(nacos_registry.stop)
        resources.push_async_callback(close_task_rag_batch_runner)
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

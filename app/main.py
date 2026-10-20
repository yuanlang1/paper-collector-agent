import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.api import run_router, setting_router, thread_router
from app.common.exceptions import register_exception_handlers
from app.config import settings
from app.database import Base, create_database
from app.grpc.server import InternalGrpcServer
from app.infrastructure.grpc.grpc_channel_pool import paper_service_grpc_channel_pool
from app.infrastructure.nacos_registry import nacos_registry
from app.rag.index_construction.base import close_index_resources
from app.rag.index_construction.memory_index_sync import (
    get_memory_index_synchronizer,
)
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

        store = await resources.enter_async_context(
            AsyncPostgresStore.from_conn_string(
                settings.DATABASE_POSTGRES_URL,
            )
        )
        await store.setup()

        app.state.checkpointer = checkpointer
        app.state.store = store

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        resources.push_async_callback(close_index_resources)
        await get_memory_index_synchronizer().ensure_collections()
        resources.push_async_callback(paper_service_grpc_channel_pool.close)
        resources.push_async_callback(close_task_rag_batch_runner)

        grpc_server = InternalGrpcServer(session_factory)
        resources.push_async_callback(grpc_server.stop)
        await grpc_server.start()
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

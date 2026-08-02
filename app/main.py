from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.config import settings
from app.api import *
from app.infrastructure.grpc_channel_pool import paper_service_grpc_channel_pool
from app.infrastructure.nacos_registry import nacos_registry
from app.core.exceptions import register_exception_handlers
from app.rag.index_construction.base import close_index_resources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

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
    await nacos_registry.start()
    try:
        yield
    finally:
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

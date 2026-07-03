from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.config import settings
from app.api import *
from app.infrastructure.nacos_registry import nacos_registry
from app.core.exceptions import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# @app.on_event("startup")
# async def on_startup():
#     init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await nacos_registry.start()
    try:
        yield
    finally:
        await nacos_registry.stop()

app = FastAPI(lifespan = lifespan)

register_exception_handlers(app)

app.include_router(agent_router)
app.include_router(settings_router)
app.include_router(ai_api_router)

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
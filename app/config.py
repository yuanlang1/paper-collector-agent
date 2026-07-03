from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

from sqlalchemy import false

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_ENV_FILE = BACKEND_DIR / ".env"
DEFAULT_SQLITE_DB_PATH = BACKEND_DIR / "literature.db"

class Settings(BaseSettings):
    """应用设置"""
    
    # 应用基本配置
    APP_NAME: str = "Literature Review System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SQL_ECHO: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 5455

    DATABASE_URL: str = (
        "mysql+pymysql://root:200366@localhost:3306/"
        "research_paper_db"
    )   


    # LLM / OpenAI 兼容API配置
    # 基础通用配置（从 .env 读取）
    # OPENAI_API_KEY: str = "sk-737aecbd7f2f46b8aa6c7190d29071ce"
    # OPENAI_BASE_URL: str = "https://api.deepseek.com"
    # OPENAI_MODEL: str = "deepseek-v4-flash"

    OPENAI_API_KEY: str = "sk-e8b60747a86040d2b33f1b83db481e13"
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OPENAI_MODEL: str = "qwen3-max-2026-01-23"

    # Nacos
    NACOS_ENABLED: bool = False
    NACOS_SERVER_ADDR: str = "127.0.0.1:8848"
    NACOS_NAMESPACE_ID: str = ""
    NACOS_GROUP_NAME: str = "DEFAULT_GROUP"
    NACOS_CLUSTER_NAME: str = "DEFAULT"
    NACOS_USERNAME: str = "nacos"
    NACOS_PASSWORD: str = "yl200366"
    NACOS_LOG_LEVEL: str = "INFO"
    NACOS_GRPC_TIMEOUT_MS: int = 5000

    # 当前 FastAPI 注册到 Nacos 的服务名
    SERVICE_NAME: str = "paper-collector-agent"
    SERVICE_IP: str = "127.0.0.1"
    SERVICE_PORT: int = 8000

    PAPER_SERVICE_NAME: str = "paper-service"
    PAPER_SERVICE_TIMEOUT_SECONDS: float = 10.0
    PAPER_SERVICE_SEARCH_TASK_CREATE_PATH: str = "/api/search/tasks"

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
DATABASE_URL = settings.DATABASE_URL
DEBUG = settings.DEBUG
SQL_ECHO = settings.SQL_ECHO
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_BASE_URL = settings.OPENAI_BASE_URL
OPENAI_MODEL = settings.OPENAI_MODEL

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from typing import List

from sqlalchemy import false

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_ENV_FILE = BACKEND_DIR / ".env"
DEFAULT_SQLITE_DB_PATH = BACKEND_DIR / "literature.db"

class Settings(BaseSettings):
    # 应用基本配置
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool
    SQL_ECHO: bool
    HOST: str
    PORT: int

    DATABASE_URL: str 

    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str
    THINKING_TYPE: str
    REASONING_EFFORT: str

    # Nacos
    NACOS_ENABLED: bool
    NACOS_SERVER_ADDR: str
    NACOS_NAMESPACE_ID: str
    NACOS_GROUP_NAME: str
    NACOS_CLUSTER_NAME: str
    NACOS_USERNAME: str
    NACOS_PASSWORD: str
    NACOS_LOG_LEVEL: str
    NACOS_LOG_DIR: str = "data/logs/nacos"
    NACOS_GRPC_TIMEOUT_MS: int

    # nacos
    SERVICE_NAME: str
    SERVICE_IP: str
    SERVICE_PORT: int

    # paper-service-rpc
    PAPER_SERVICE_GRPC_NAME: str
    PAPER_SERVICE_GRPC_TIMEOUT_SECONDS: float
    PAPER_SERVICE_GRPC_MAX_RECEIVE_MESSAGE_LENGTH: int

    SERPAPI_SEARCH_URL: str
    SERPAPI_API_KEY: str
    ARXIV_API_URL: str
    DBLP_API_URL: str
    CROSSREF_API_URL: str
    CROSSREF_MAILTO: str = ""
    CROSSREF_TIMEOUT_SECONDS: float = 15.0

    EASY_SCHOLAR_URL: str
    EASY_SCHOLAR_SECRET_KEY: str 
    EASY_SCHOLAR_TIMEOUT_SECONDS: float

    ARTIFACT_BASE_DIR: str

    PAPER_SEARCH_TARGET_PAPER_COUNT: int = 30
    PAPER_SEARCH_MAX_SUPPLEMENTAL_ROUNDS: int = 2
    PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE: int = 2

    PAPER_SEARCH_ARXIV_MAX_PAGES: int = 2
    PAPER_SEARCH_ARXIV_PAGE_SIZE: int = 50
    PAPER_SEARCH_ARXIV_TOTAL_LIMIT: int = 10

    PAPER_SEARCH_DBLP_MAX_PAGES: int = 2
    PAPER_SEARCH_DBLP_PAGE_SIZE: int = 50
    PAPER_SEARCH_DBLP_TOTAL_LIMIT: int = 10
    
    PAPER_SEARCH_GOOGLE_SCHOLAR_MAX_PAGES: int = 2
    PAPER_SEARCH_GOOGLE_SCHOLAR_PAGE_SIZE: int = 10
    PAPER_SEARCH_GOOGLE_SCHOLAR_TOTAL_LIMIT: int = 20

    LANGSMITH_TRACING: bool
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    # mineru settings
    MINERU_SINGLE_TASK_PATH: str
    MINERU_BATCH_TASK_PATH: str
    MINERU_BATCH_RESULT_PATH: str
    MINERU_TOKEN: str
    MINERU_MAX_BATCH_SIZE: int = 50
    MINERU_MAX_PARALLEL_BATCHES: int = 2
    MINERU_BASE_URL: str

    # embedding settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    PAPER_COLLECTION_NAME: str
    PAPER_CONTENT_COLLECTION_NAME: str
    DENSE_EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    SPARSE_EMBEDDING_MODEL_NAME: str = "Qdrant/bm25"
    EMBEDDING_DEVICE: str = "cpu"

    @model_validator(mode="after")
    def validate_paper_search_pagination(self) -> "Settings":
        for source in ("ARXIV", "DBLP", "GOOGLE_SCHOLAR"):
            page_size = getattr(self, f"PAPER_SEARCH_{source}_PAGE_SIZE")
            max_pages = getattr(self, f"PAPER_SEARCH_{source}_MAX_PAGES")
            total_limit = getattr(self, f"PAPER_SEARCH_{source}_TOTAL_LIMIT")
            if total_limit > page_size * max_pages:
                raise ValueError(
                    f"PAPER_SEARCH_{source}_TOTAL_LIMIT cannot exceed "
                    f"PAGE_SIZE * MAX_PAGES"
                )
        return self

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
os.environ["LANGSMITH_TRACING"] = str(settings.LANGSMITH_TRACING).lower()
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
DATABASE_URL = settings.DATABASE_URL
DEBUG = settings.DEBUG
SQL_ECHO = settings.SQL_ECHO
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_BASE_URL = settings.OPENAI_BASE_URL
OPENAI_MODEL = settings.OPENAI_MODEL

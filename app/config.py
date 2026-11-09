import os
from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    VENUE_CACHE_TTL_SECONDS: int = Field(default=86400, gt=0)
    REDIS_URL: str = Field(default="", repr=False)
    REDIS_CONNECT_TIMEOUT_SECONDS: float = Field(default=1.0, gt=0)
    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(default=1.0, gt=0)
    REDIS_MAX_CONNECTIONS: int = Field(default=20, gt=0)
    REDIS_KEY_PREFIX: str = Field(
        default="paper-collector", min_length=1, pattern=r"\S"
    )

    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    LLM_PROVIDER: str
    OPENAI_MODEL: str
    THINKING_TYPE: str
    REASONING_EFFORT: str

    SERVICE_NAME: str

    # paper-service internal HTTP API
    PAPER_SERVICE_BASE_URL: AnyHttpUrl
    PAPER_SERVICE_CONNECT_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    PAPER_SERVICE_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)

    SERPAPI_SEARCH_URL: str
    SERPAPI_API_KEY: str = ""
    ARXIV_API_URL: str
    DBLP_API_URL: str
    CROSSREF_API_URL: str
    CROSSREF_MAILTO: str = ""
    CROSSREF_TIMEOUT_SECONDS: float = 15.0
    PAPER_CROSSREF_TASK_TIMEOUT_SECONDS: float = Field(default=100, gt=0)
    PAPER_CROSSREF_GROUP_TIMEOUT_SECONDS: float = Field(default=300, gt=0)

    FIRECRAWL_API_KEY: str = ""
    FIRECRAWL_SEARCH_URL: str = "https://api.firecrawl.dev/v2/search"
    FIRECRAWL_TIMEOUT_SECONDS: float = 30.0

    EASY_SCHOLAR_URL: str
    EASY_SCHOLAR_SECRET_KEY: str
    EASY_SCHOLAR_TIMEOUT_SECONDS: float

    ARTIFACT_BASE_DIR: str

    OSS_ENABLED: bool = True
    OSS_BUCKET: str
    OSS_REGION: str
    OSS_ENDPOINT: str
    OSS_PREFIX: str
    OSS_ACCESS_KEY_ID: str
    OSS_ACCESS_KEY_SECRET: str
    OSS_SESSION_TOKEN: str | None = None

    PAPER_SEARCH_TARGET_PAPER_COUNT: int = 30
    PAPER_SEARCH_MAX_SUPPLEMENTAL_ROUNDS: int = 2
    PAPER_SEARCH_MAX_EXTRA_PAGES_PER_SOURCE: int = 2

    LANGSMITH_TRACING: bool
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    # mineru settings
    MINERU_SINGLE_TASK_PATH: str
    MINERU_BATCH_TASK_PATH: str
    MINERU_BATCH_RESULT_PATH: str
    MINERU_TOKEN: str
    MINERU_BASE_URL: str
    MINERU_PARSE_TIMEOUT_SECONDS: int = 1200
    TASK_RAG_POST_PROCESSING_GRACE_SECONDS: int = 600
    TASK_RAG_WORKFLOW_TIMEOUT_SECONDS: int = Field(default=1800, ge=1, le=3600)
    PAPER_ABSTRACT_TASK_TIMEOUT_SECONDS: int = Field(default=180, gt=0)
    PAPER_ABSTRACT_GROUP_TIMEOUT_SECONDS: int = Field(default=900, gt=0)
    SUBAGENT_DISPATCH_TIMEOUT_SECONDS: int = 1800
    TASK_REVIEW_SUBAGENT_TIMEOUT_SECONDS: int = 3600
    TASK_REVIEW_MAX_REFLECTION_ROUNDS: int = Field(default=5, ge=1, le=5)
    TASK_REVIEW_MAX_WRITING_REVISION_ROUNDS: int = Field(default=5, ge=0, le=5)
    SUBAGENT_DISPATCH_TIMEOUT_GRACE_SECONDS: int = 60

    # embedding settings
    DASHSCOPE_API_KEY: SecretStr = Field(min_length=1)
    QWEN_EMBEDDING_BASE_URL: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    QWEN_EMBEDDING_MODEL: str = "qwen3.7-text-embedding-flash"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    PAPER_COLLECTION_NAME: str
    PAPER_CONTENT_COLLECTION_NAME: str
    MODEL_DIR: str = "models"
    SPARSE_EMBEDDING_MODEL_NAME: str = "Qdrant/bm25"
    EMBEDDING_DEVICE: str = "cpu"

    LANGGRAPH_CHECKPOINT_PATH: str = "data/langgraph-checkpoints.db"
    CHAT_HISTORY_DB_PATH: str = "data/chat-history.db"

    LLM_PROVIDER: str = "openai"
    APP_TIMEZONE: str = "Asia/Shanghai"
    AGENT_SKILLS_DIR: str = "data/agent-skills"
    AGENT_HISTORY_TURNS: int = 12

    DATABASE_POSTGRES_URL: str

    @model_validator(mode="after")
    def validate_task_rag_timeout(self) -> "Settings":
        minimum = (
            self.MINERU_PARSE_TIMEOUT_SECONDS
            + self.TASK_RAG_POST_PROCESSING_GRACE_SECONDS
        )
        if self.TASK_RAG_WORKFLOW_TIMEOUT_SECONDS < minimum:
            raise ValueError(
                "TASK_RAG_WORKFLOW_TIMEOUT_SECONDS must cover parsing and post-processing"
            )
        return self

    @model_validator(mode="after")
    def validate_agent_history(self) -> "Settings":
        if self.AGENT_HISTORY_TURNS < 1:
            raise ValueError("AGENT_HISTORY_TURNS must be at least 1")
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
DEBUG = settings.DEBUG
SQL_ECHO = settings.SQL_ECHO
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_BASE_URL = settings.OPENAI_BASE_URL
OPENAI_MODEL = settings.OPENAI_MODEL

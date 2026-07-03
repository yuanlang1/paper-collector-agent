from sqlalchemy import create_engine, text
import logging
import app.config as settings
from typing import Generator
from sqlalchemy.orm import Session, sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.SQL_ECHO,
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
# 创建基类
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话
    用于FastAPI的依赖注入
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# def init_db():
#     """
#     初始化数据库
#     创建所有表 + 执行幂等迁移 + 回填旧数据
#     """
#     # 导入所有模型以确保它们被注册
#     from app import models  # noqa: F401
    
#     # 创建所有表（包括新增的 api_usage_logs）
#     Base.metadata.create_all(bind=engine)
#     print("✅ 数据库表创建成功！")

#     logger.info("✅ api_usage_logs 表已就绪（由 create_all 自动管理）")

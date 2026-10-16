import app.config as settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_database():
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.SQL_ECHO,
    )
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    return engine, session_factory
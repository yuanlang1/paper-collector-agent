from app.config import settings
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker


def create_database():
    engine = create_engine(
        make_url(settings.DATABASE_POSTGRES_URL).set(drivername="postgresql+psycopg"),
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

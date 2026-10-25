from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import Pool

from app.config import settings

Base = declarative_base()


def create_database(
    *, poolclass: type[Pool] | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    url = make_url(settings.DATABASE_POSTGRES_URL).set(drivername="postgresql+psycopg")
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.SQL_ECHO,
        **({"poolclass": poolclass} if poolclass is not None else {}),
    )
    session_factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    return engine, session_factory


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session

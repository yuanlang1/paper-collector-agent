from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.agent_runner import AgentRunner


async def chat(
    *,
    db: AsyncSession,
    conversation_id: str,
    user_id: str,
    message: str,
    checkpointer,
    store,
) -> AsyncIterator[dict[str, Any]]:
    runner = AgentRunner(
        session_id = conversation_id,
        run_id = f"run_{uuid4().hex}",
        user_id = user_id,
        checkpointer = checkpointer,
        store = store,
    )

    async for event in runner.run(db, message):
            yield event
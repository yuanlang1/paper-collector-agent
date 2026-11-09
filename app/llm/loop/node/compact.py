from langgraph.runtime import Runtime

from app.llm.context import BaseAgentContext
from app.llm.loop.state import LoopState
from app.services import agent_thread_summary_service


class CompactNode:
    async def __call__(
        self, state: LoopState, runtime: Runtime[BaseAgentContext]
    ) -> dict:
        context = runtime.context
        result = await agent_thread_summary_service.compact(
            user_id=context.user_id,
            thread_id=context.thread_id,
            messages=state["messages"],
            keep_last_n_turns=context.keep_last_n_turns,
        )
        if isinstance(result, dict):
            runtime.stream_writer(
                {"type": "compact", "status": "failed", "message": "压缩失败"}
            )
        elif result is not None:
            runtime.stream_writer(
                {"type": "compact", "status": "success", "message": "压缩成功"}
            )
        return {}

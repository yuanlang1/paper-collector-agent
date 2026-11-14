from langgraph.runtime import Runtime

from app.llm.context import BaseAgentContext
from app.llm.loop.state import LoopState
from app.services import agent_thread_summary_service


class CompactNode:
    async def __call__(
        self, state: LoopState, runtime: Runtime[BaseAgentContext]
    ) -> dict:
        context = runtime.context
        writer = runtime.stream_writer

        writer({"type": "compact_start", "status": "running", "message": "开始压缩"})

        result = await agent_thread_summary_service.compact(
            user_id=context.user_id,
            thread_id=context.thread_id,
            messages=state["messages"],
            keep_last_n_turns=context.keep_last_n_turns,
        )

        if isinstance(result, dict):
            writer(
                {"type": "compact_failed", "status": "failed", "message": "压缩失败"}
            )

        elif result is None:
            writer({"type": "compact_end", "status": "skipped", "message": "跳过压缩"})

        else:
            writer({"type": "compact_end", "status": "success", "message": "压缩完成"})
        return {}

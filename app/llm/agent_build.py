from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai.types.shared import ReasoningEffort
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.subagents.paper_search import build_paper_search_tool
from app.llm.subagents.task_indexing import task_indexing_agent
from app.llm.subagents.task_review import task_review_agent
from app.llm.tools.agent_tools.subagent import build_subagent_tool
from app.llm.tools.registry import ToolRegistry
from app.llm.tools.web_tools.search_web import search_web_tool
from app.services import llm_setting_service


def build_tools(db: AsyncSession | None, user_id: int, *, model: BaseChatModel) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_paper_search_tool(model=model), "论文检索子代理")
    registry.register(task_indexing_agent, "索引子代理")
    registry.register(task_review_agent, "综述生成子代理")
    registry.register(search_web_tool, "网络搜索")
    registry.register(build_subagent_tool(model=model), "子代理")

    return registry


async def get_model(
    db: AsyncSession,
    model_id: int,
    user_id: int,
    *,
    think_effort: ReasoningEffort = None,
) -> BaseChatModel:
    llm_config = await llm_setting_service.get_llm_config(
        db,
        model_id,
        user_id=user_id,
    )

    return ChatOpenAI(
        model=llm_config.model,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        streaming=True,
        reasoning_effort=think_effort,
    )

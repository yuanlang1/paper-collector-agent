from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai.types.shared import ReasoningEffort
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.subagents.paper_search import paper_search_agent
from app.llm.subagents.task_indexing import task_indexing_agent
from app.llm.subagents.task_review import task_review_agent
from app.llm.tools.registry import ToolRegistry
from app.llm.tools.web_tools.search_web import search_web_tool
from app.services import llm_setting_service


def build_tools(db: AsyncSession, user_id: int) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(paper_search_agent, "论文检索子代理")
    registry.register(task_indexing_agent, "索引子代理")
    registry.register(task_review_agent, "综述生成子代理")
    registry.register(search_web_tool, "网络搜索")

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

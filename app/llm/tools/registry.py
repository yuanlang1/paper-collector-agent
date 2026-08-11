from app.llm.tools.base import AppToolSpec
from app.llm.tools.file_tools.download_file import DOWNLOAD_FILE_TOOL
from app.llm.tools.search_tools.arxiv.search_arxiv import ARXIV_SEARCH_TOOL
from app.llm.tools.search_tools.dblp.search_dblp import DBLP_SEARCH_TOOL
from app.llm.tools.search_tools.crossref.search_crossref import (
    CROSSREF_SEARCH_TOOL,
)
from app.llm.tools.search_tools.google_scholar.search_google_scholar import GOOGLE_SCHOLAR_SEARCH_TOOL
from app.llm.tools.task_tools.search_task.search_task_create import ADD_QUERY_TASK_TOOL
from app.llm.tools.task_tools.search_task.rag_status import GET_TASK_RAG_STATUS_TOOL
from app.llm.tools.task_tools.search_task.start_task_rag import (
    START_TASK_RAG_TOOL,
)
from app.llm.tools.task_tools.search_task.start_task_rag import (
    START_TASK_RAG_TOOL,
)
from app.llm.tools.venue_tools.easy_scholar import EASY_SCHOLAR_VENUE_TOOL


ALL_TOOLS: list[AppToolSpec] = [
    ARXIV_SEARCH_TOOL,
    DBLP_SEARCH_TOOL,
    CROSSREF_SEARCH_TOOL,
    GOOGLE_SCHOLAR_SEARCH_TOOL,
    ADD_QUERY_TASK_TOOL,
    GET_TASK_RAG_STATUS_TOOL,
    START_TASK_RAG_TOOL,
    EASY_SCHOLAR_VENUE_TOOL,
    DOWNLOAD_FILE_TOOL
]

TOOL_BY_NAME: dict[str, AppToolSpec] = {
    tool.name: tool for tool in ALL_TOOLS
}

def get_tool_specs(
        allowed_tools: list[str] | None = None
    ) -> list[AppToolSpec]:
    if allowed_tools is None:
        return ALL_TOOLS

    allowed = set(allowed_tools)
    return [tool for tool in ALL_TOOLS if tool.name in allowed]

def tool_descriptions() -> str:
    return "\n".join(
        f"- tool name: {tool.name}: 描述: {tool.description}, 参数：{tool.args_schema.model_json_schema()}"
        for tool in ALL_TOOLS
    )

def requires_confirmation(
        tool_names: list[str]
    ) -> bool:
    return any(
        TOOL_BY_NAME[name].requires_confirmation
        for name in tool_names
        if name in TOOL_BY_NAME
    )

    

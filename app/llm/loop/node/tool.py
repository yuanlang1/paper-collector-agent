import httpx
from langchain_core.tools import ToolException
from langgraph.prebuilt import ToolNode


def handler_tool_error(exc: ToolException | httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"工具请求失败，HTTP {exc.response.status_code}。"
    if isinstance(exc, httpx.HTTPError):
        return "工具网络请求失败，请稍后重试。"
    return str(exc) or "工具执行失败"


def build_tool_node(tools):
    return ToolNode(
        tools=tools,
        handle_tool_errors=handler_tool_error,
    )

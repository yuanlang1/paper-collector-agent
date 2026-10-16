
import logging
from langchain.tools import ToolException
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

def handler_tool_error(exc: ToolException) -> str:
    return "工具执行失败"

def wrap_tool_call(request, execute):
    tool_call_id = request.runtime.tool_call_id,
    tool_name = request.runtime.tool_name, 
    attempts = request.runtime.context.max_attempts
    
    for i in range(attempts):
        try:
            tool_msg = execute(request)
            break
        except ConnectionError as e:
            logger.info(
                f"工具执行失败: %s，当前调用次数：%s, 调用上限 %s, 异常消息：%s",
                tool_name, i, attempts
            )

    if not tool_msg:
        tool_msg = ToolMessage(
            tool_call_id = tool_call_id,
            content = "调用次数达到上限，调用失败"
        )

    return tool_msg


def build_tool_node(tools):
    return ToolNode(
        tools = tools,
        handle_tool_errors = handler_tool_error,
        wrap_tool_call = wrap_tool_call
    )
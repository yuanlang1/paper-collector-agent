import json
from typing import Callable, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.system import SystemMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import Runnable
from langgraph.types import interrupt
from app.llm.graph.state import AgentState
from app.llm.prompt import ROUTER_SYSTEM_PROMPT, UNIFIED_AGENT_SYSTEM_PROMPT
from app.llm.tools.registry import tool_descriptions

def _db_from_config(config: RunnableConfig) -> Any:
    return (config.get("configurable") or {}).get("db") 

def _content_to_text(content: Any) -> Any:
    if isinstance(content, str):
        return content
    
    return json.dumps(content, ensure_ascii=False, default=str)

def _tool_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    
    return json.dumps(value, ensure_ascii=False, default=str)

class AgentGraphNodes:
    def __init__(
        self,
        *,
        router,
        ask_llm,
        agent_llm,
        tool_factory,
        build_system_prompt: Callable[[str, Any], str],
    ) -> None:
        self.router = router
        self.ask_llm = ask_llm
        self.agent_llm = agent_llm
        self.tool_factory = tool_factory
        self.build_system_prompt = build_system_prompt

    async def route_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        db = _db_from_config(config)
        decision = await self.router.ainvoke([
                SystemMessage(content = self.build_system_prompt(ROUTER_SYSTEM_PROMPT, db)),
                HumanMessage(content = f"""
                    可用工具：
                    {tool_descriptions()}

                    用户最新消息：
                    {state.message}

                    请判断 route、allowed_tools、requires_confirmation。
                    """),
            ]
        )

        allowed_tools = (
            []
            if decision.route == "ask"
            else [getattr(tool, "value", tool) for tool in decision.allowed_tools]
        )

        return {
            "route": decision.route,
            "allowed_tools": allowed_tools,
            "requires_confirmation": decision.requires_confirmation,
            "reasoning": decision.reasoning,
            "workflow": [*state.workflow, "route"],
        }

    async def ask_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        db = _db_from_config(config)

        messages = state.messages or [HumanMessage(content=state.message)]
        result = await self.ask_llm.ainvoke([
            SystemMessage(content = self.build_system_prompt(UNIFIED_AGENT_SYSTEM_PROMPT, db)),
            *messages
        ])

        return {
            "messages": [result],
            "reply": _content_to_text(result.content),
            "workflow": [*state.workflow, "ask"],
        }
    
    async def agent_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        db = _db_from_config(config)

        tools = self.tool_factory.build(
            db = db,
            allowed_tools = state.allowed_tools,
        )

        model = self.agent_llm.bind_tools(tools) if tools else self.agent_llm
        messages = state.messages or [SystemMessage(content=state.message)]

        result = await model.ainvoke([
            SystemMessage(content = self.build_system_prompt(UNIFIED_AGENT_SYSTEM_PROMPT, db)),
            *messages,
        ])

        return {
            "messages": [result],
            "workflow": [*state.workflow, "agent"],
        }
    
    async def tool_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        db = _db_from_config(config)

        if not state.messages:
            return {}
        
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}
        
        tools = self.tool_factory.build(
            db = db,
            allowed_tools = state.allowed_tools,
        )
        tool_by_name = {tool.name: tool for tool in tools}

        tool_message = []
        actions = []

        for call in last_message.tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args") or {}
            tool_call_id = call.get("id")

            if tool_name not in tool_by_name:
                result = {
                    "ok": False,
                    "message": f"工具不可用或不被允许: {tool_name}"
                }
            else:
                try:
                    result = await tool_by_name[tool_name].ainvoke(tool_args)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "message": str(exc)
                    }

                actions.append({
                    "tool": tool_name,
                    "params": tool_args,
                    "result": result,
                    "success": bool(result.get("ok", True)) if isinstance(result, dict) else True,
                    "error": result.get("message") if isinstance(result, dict) and result.get("ok") is False else None,
                })

                tool_message.append(
                    ToolMessage(
                        content = _tool_content(result),
                        tool_call_id = tool_call_id,
                        name = tool_name,
                    )
                )

        return {
            "messages": tool_message,
            "actions": [*state.actions, *actions],
            "workflow": [*state.workflow, "tool"],
        }

    async def human_approval_node(
        self, 
        state: AgentState
    ) -> dict:
        last_message = state.messages[-1]

        tool_calls = []
        if isinstance(last_message, AIMessage):
            tool_calls = last_message.tool_calls or []
        
        payload = {
            "type": "tool_approval",
            "message": "是否允许执行以下工具调用？",
            "conversation_id": state.conversation_id,
            "route": state.route,
            "reasoning": state.reasoning,
            "allowed_tools": state.allowed_tools,
            "tool_calls": [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                }
                for call in tool_calls
            ],
        }

        decision = interrupt(payload)
    
        approved = False
        if isinstance(decision, bool):
            approved = decision
        elif isinstance(decision, dict):
            approved = decision.get("approved") is True
        
        if not approved:
            return {
                "human_decision": "rejected",
                "interrupt_payload": payload,
                "reply":"操作已取消，未执行工具。",
                "status": "rejected",
                "workflow": [*state.workflow, "human_approval"] 
            }
        
        return {
            "human_decision": "approved",
            "interrupt_payload": payload,
            "workflow": [*state.workflow, "human_approval"]
        }

    async def final_node(
        self, 
        state: AgentState
    ) -> dict:
        if state.requires_confirmation:
            return {
                "reply": "这个操作会修改系统数据或启动较重任务。请确认后再执行。",
                "status": "confirmation_required",
                "actions": state.actions,
                "workflow": [*state.workflow, "final"],
            }
        
        if state.actions:
            status = "tool_executed"
        else:
            status = "completed"

        if state.reply:
            return {
                "status": status,
                "actions": state.actions,
                "workflow": [*state.workflow, "final"],
            }

        for message in reversed(state.messages):
            if isinstance(message, AIMessage) and message.content:
                return {
                    "reply": _content_to_text(message.content),
                    "status": status,
                    "actions": state.actions,
                    "workflow": [*state.workflow, "final"],
                }

        return {
            "reply": "",
            "status": status,
            "actions": state.actions,
            "workflow": [*state.workflow, "route"],
        }
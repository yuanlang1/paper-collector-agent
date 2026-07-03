from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import BaseMessage
from typing import Literal

from pydantic import ConfigDict
from typing import Any
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    message: str = Field(..., description="The message to be sent to the agent")
    conversation_id: str = Field(..., description="The id of the conversation")

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    route: Literal["ask", "agent"] | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    reasoning: str | None = None

    reply: str | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)

    status: Literal[
            "completed",
            "confirmation_required",
            "tool_executed",
            "failed",
            "rejected",
    ] | None = None

    workflow: list[str] = Field(default_factory=list)
    error: str | None = None

    human_decision: Literal["approved", "rejected"] | None = None
    interrupt_payload: dict[str, Any] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
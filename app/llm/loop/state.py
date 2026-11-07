from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class SummaryItem(TypedDict):
    content: str
    start_id: str
    end_id: str
    created_at: str


class LoopState(MessagesState):
    summaries: Annotated[list[SummaryItem], add]

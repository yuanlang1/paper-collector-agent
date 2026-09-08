"""Compatibility imports for the paper-search workflow schemas."""

from app.llm.graph.workflows.paper_search_schemas import (
    NonBlankString,
    PaperTypeCode,
    PromptIntent,
    PromptUnderstandingArgs,
    SearchTagArgs,
    SourceTypeValue,
)
from app.llm.tools.task_tools.search_task.rag_status import GetTaskRagStatusArgs
from app.llm.tools.task_tools.search_task.search_task_create import AddQueryTaskArgs

__all__ = [
    "AddQueryTaskArgs",
    "GetTaskRagStatusArgs",
    "NonBlankString",
    "PaperTypeCode",
    "PromptIntent",
    "PromptUnderstandingArgs",
    "SearchTagArgs",
    "SourceTypeValue",
]

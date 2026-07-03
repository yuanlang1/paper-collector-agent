from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    search_papers = "search_papers"
    list_staging = "list_staging"
    promote_papers = "promote_papers"
    delete_staging = "delete_staging"
    search_library = "search_library"
    sync_citations = "sync_citations"
    system_status = "system_status"
    generate_framework = "generate_framework"
    start_review_task = "start_review_task"
    run_phd_pipeline = "run_phd_pipeline"
    list_reviews = "list_reviews"
    export_review = "export_review"
    semantic_search = "semantic_search"
    manage_groups = "manage_groups"
    check_task_progress = "check_task_progress"
    modify_task_requirements = "modify_task_requirements"
    configure_discipline = "configure_discipline"
    download_pdf = "download_pdf"
    screen_papers = "screen_papers"
    enrich_papers = "enrich_papers"
    prisma_stage = "prisma_stage"
    institutional_login = "institutional_login"


class RouteDecision(BaseModel):
    route: Literal["ask", "agent"] = Field(
        description="ask=普通聊天/学术问答；agent=需要调用系统工具"
    )
    allowed_tools: List[ToolName] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False)
    reasoning: str = Field(description="中文说明路由原因")

class AgentResponse(BaseModel):
    reply: str
    action: Dict[str, Any] | None = None
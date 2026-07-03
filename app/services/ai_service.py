from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.core.response import ServiceResponse
from app.llm.langchain_service import get_agent_service
from app.llm.prompt import *

class AIAbstractResponse(BaseModel):
    aiAbstract: str

class AIService:
    def __init__(self) -> None:
        pass

    async def get_ai_abstract(
        self,
        abstract: str,
    ) -> ServiceResponse[AIAbstractResponse]:
        service = get_agent_service()

        result = await service.ask(
            system_message = SystemMessage(AI_ABSTRACT_SYSTEM_PROMPT),
            human_message = HumanMessage(AI_ABSTRACT_HUMAN_PROMPT.format(abstract = abstract))
        )

        return ServiceResponse.build_success_response(
            AIAbstractResponse(
                aiAbstract=result["reply"]
            )
        )

_ai_service: AIService | None = None
    
def get_ai_service() -> AIService:
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service
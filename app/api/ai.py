from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.api.agent import router
from app.core.response import ServiceResponse
from app.services.ai_service import AIAbstractResponse, AIService, get_ai_service

router = APIRouter(prefix="/api/ai", tags=["ai"])

class PostAbstractRequest(BaseModel):
    paperAbstract: str = Field(..., description = "摘要")


@router.post("/ai_abstract")
async def get_ai_abstract(
    payload: PostAbstractRequest,
    service: AIService = Depends(get_ai_service),
) -> ServiceResponse[AIAbstractResponse]:
    """
    Get ai abstract from abstract
    """
    return await service.get_ai_abstract(abstract = payload.paperAbstract)


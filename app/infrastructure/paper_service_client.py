from typing import Any, Dict

import httpx

from app.config import settings
from app.infrastructure.nacos_registry import nacos_registry


def _get_instance_attr(
    instance: Any,
    name: str,
    default: Any = None
) -> Any:
    if isinstance(instance, dict):
        return instance.get(name, default)
    
    return getattr(instance, name, default)

def _build_service_base_url(
    instance: Any
) -> str:
    ip = _get_instance_attr(instance, "ip")
    port = _get_instance_attr(instance, "port")
    metadata = _get_instance_attr(instance, "metadata", {}) or {}

    if not ip or not port:
        raise RuntimeError("Nacos 返回的 paper-service 实例缺少 ip 或 port")

    scheme = metadata.get("scheme") or metadata.get("protocol") or "http"
    context_path = metadata.get("contextPath") or metadata.get("context_path") or ""
    context_path = f"/{context_path.strip('/')}" if context_path else ""

    return f"{scheme}://{ip}:{port}{context_path}"

def _join_url(
    base_url: str, 
    path: str
) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

class PaperServiceClient:
    async def _base_url(self) -> str:
        instance = await nacos_registry.choose_instance(settings.PAPER_SERVICE_NAME)
        return _build_service_base_url(instance)
    
    async def create_search_task(
        self,
        search_understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        base_url = await self._base_url()
        url = _join_url(
            base_url,
            settings.PAPER_SERVICE_SEARCH_TASK_CREATE_PATH
        )

        payload = {
            "task_type": "search_task_create",
            "search_understanding": search_understanding,
        }

        async with httpx.AsyncClient(
            timeout = settings.PAPER_SERVICE_TIMEOUT_SECONDS
        ) as client:
            response = await client.post(url, json = payload)   
        
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "ok": False,
                "message": "paper-service 创建搜索任务失败",
                "status_code": exc.response.status_code,
                "response": exc.response.text,
                "search_understanding": search_understanding,
            }

        try:
            result = response.json()
        except ValueError:
            result = {"raw": response.text}
        
        return {
            "ok": True,
            "message": "搜索任务已创建",
            "paper_service": {
                "service_name": settings.PAPER_SERVICE_NAME,
                "url": url,
            },
            "search_understanding": search_understanding,
            "result": result,
        }

paper_service_client = PaperServiceClient()
from typing import Any

import httpx

from app.config import settings
from app.infrastructure.http import http_manager


class PaperServiceError(RuntimeError):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class PaperServiceHttpClient:
    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self.http = http

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        http = self.http if self.http is not None else http_manager.paper_service_client

        base_url = str(settings.PAPER_SERVICE_BASE_URL).rstrip("/")

        response = await http.request(method, base_url + path, **kwargs)
        response.raise_for_status()
        body = object_value(response.json())
        if type(body.get("success")) is not bool or type(body.get("code")) is not int:
            raise ValueError("Invalid paper-service response envelope")
        if not body["success"] or body["code"] != 0:
            raise PaperServiceError(
                body.get("message") or "Paper service request failed", body["code"]
            )
        return body


paper_service_http_client = PaperServiceHttpClient()

CLIENT_ERRORS = (httpx.HTTPError, KeyError, TypeError, ValueError, RuntimeError)


def object_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected a paper-service JSON object")
    return value


def list_value(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("Expected a paper-service JSON array")
    return value


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Expected a paper-service JSON string")
    return value


def positive_id(value: Any, field: str = "task_id") -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


class ServiceClient:
    def __init__(self, http: PaperServiceHttpClient | None = None) -> None:
        self.http = http or paper_service_http_client

    @staticmethod
    def _success(result: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "result": result,
            "error": None,
            "metadata": {"source": "paper_service_http", "code": response["code"]},
        }

    @staticmethod
    def _error(exc: Exception) -> dict[str, Any]:
        metadata: dict[str, Any] = {"source": "paper_service_http"}
        if isinstance(exc, PaperServiceError):
            metadata["code"] = exc.code
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"Paper service HTTP status {exc.response.status_code}"
            metadata["http_status"] = exc.response.status_code
        elif isinstance(exc, httpx.RequestError):
            error = f"Paper service HTTP request failed: {type(exc).__name__}"
        else:
            error = str(exc)
        return {"ok": False, "result": None, "error": error, "metadata": metadata}

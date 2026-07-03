import logging
from typing import Any

from fastapi import FastAPI, Request, status
from requests import status_codes
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.response import ServiceResponse

logger = logging.getLogger(__name__)

class AppException(Exception):
    def __init__(
        self, 
        message: str,
        code: int = 1,
        status_code:  int = status.HTTP_400_BAD_REQUEST,
        other: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        self.other = other
        super().__init__(message)

class BadRequestException(AppException):
    def __init__(
        self,
        message: str = "请求参数错误",
    ) -> None:
        super().__init__(
            message = message,
            code = 1,
            status_code = status.HTTP_400_BAD_REQUEST,
        )

class UnauthorizedException(AppException):
    def __init__(
        self, 
        message: str = "未授权",
    ) -> None:
        super().__init__(
            message = message,
            code = 1,
            status_code = status.HTTP_401_UNAUTHORIZED,
        )

class ForbiddenException(AppException):
    def __init__(
        self, 
        message: str = "无权限访问",
    ) -> None:
        super().__init__(
            message = message,
            code = 1,
            status_code = status.HTTP_403_FORBIDDEN,
        )

class NotFoundException(AppException):
    def __init__(
        self, 
        message: str = "资源不存在",
    ) -> None:
        super().__init__(
            message = message,
            code = 1,
            status_code = status.HTTP_404_NOT_FOUND,
        )

class ServiceUnavailableException(AppException):
    def __init__(
        self, 
        message: str = "服务暂不可用",
    ) -> None:
        super().__init__(
            message = message,
            code = 1,
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
        )

# 返回json格式 dict
def _build_error_response(
    *,
    code: int,
    message: str,
    status_code: int,
    other: dict[str, Any] | None = None,
) -> JSONResponse:
    response = ServiceResponse.build_error_response(
        code = code,
        message = message,
    )

    if other:
        response.other = other

    return JSONResponse(
        status_code = status_code,
        content = response.model_dump(),
    )


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    return _build_error_response(
        code = exc.code,
        message = exc.message,
        status_code = exc.status_code,
        other = exc.other,
    )

async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    return _build_error_response(
        code = exc.status_code,
        message = str(exc.detail),
        status_code = exc.status_code,
    )

async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    first_error = errors[0] if errors else {}

    loc = first_error.get("loc", [])
    field = ".".join(str(item) for item in loc if item != "body")
    message = first_error.get("msg", "请求参数校验失败")

    if field:
        message = f"{field}: {message}"

    return _build_error_response(
        code = 1,
        message = message,
        status_code = status.HTTP_400_BAD_REQUEST,
        other = {
            "errors": errors,
        },
    )

async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    return _build_error_response(
        code = 1,
        message = "数据校验失败",
        status_code = status.HTTP_400_BAD_REQUEST,
        other = {
            "errors": exc.errors(),
        },
    )

async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)

    return _build_error_response(
        code = 1,
        message = "服务器内部错误",
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
    )

def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
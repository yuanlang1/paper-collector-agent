from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class ServiceResponse(BaseModel, Generic[T]):
    SUCCESS_CODE: ClassVar[int] = 0

    code: int = Field(default=0)
    success: bool = Field(default=True)

    data: T | None = None
    message: str = ""

    other: dict[str, Any] | None = None

    @classmethod
    def build_success_response(
        cls,
        data: T | None = None,
        message: str = "",
    ) -> "ServiceResponse[T]":
        return cls(
            code=cls.SUCCESS_CODE,
            success=True,
            data=data,
            message=message,
            other=None,
        )

    @classmethod
    def build_error_response(
        cls,
        code: int,
        message: str,
        *,
        other: dict[str, Any] | None = None,
    ) -> "ServiceResponse[T]":
        return cls(
            code=code,
            success=False,
            data=None,
            message=message,
            other=other,
        )

    def is_success(self) -> bool:
        return (
            self.success
            and self.code == self.SUCCESS_CODE
        )

    def is_error(self) -> bool:
        return not self.is_success()

    def has_data(self) -> bool:
        return self.data is not None

    def has_message(self) -> bool:
        return bool(self.message.strip())

    def has_extra(self) -> bool:
        return bool(self.other)

    def has_extra_key(
        self,
        key: str,
    ) -> bool:
        return (
            self.other is not None
            and key in self.other
        )

    def get_extra(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        if self.other is None:
            return default

        return self.other.get(key, default)

    def put_extra(
        self,
        key: str,
        value: Any,
    ) -> None:
        if self.other is None:
            self.other = {}

        self.other[key] = value

    def remove_extra(
        self,
        key: str,
    ) -> Any:
        if self.other is None:
            return None

        return self.other.pop(key, None)

    def clear_extra(self) -> None:
        self.other = None
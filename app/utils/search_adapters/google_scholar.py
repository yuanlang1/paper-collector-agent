from collections.abc import Mapping
from typing import Any

import httpx
from langchain_core.tools import ToolException

from app.llm.tools.search_tools.google_scholar.search_google_scholar import (
    GoogleScholarSearchArgs,
    google_scholar_search_service,
)
from app.utils.search_adapters.contracts import (
    SearchBatchResult,
    batch_result,
    failed_result,
    source_arguments,
)


class GoogleScholarAdapter:
    __slots__ = ()
    provider_id = "google_scholar"

    async def search(
        self,
        *,
        query_args: Mapping[str, Any],
        settings: Mapping[str, Any],
        credentials: Mapping[str, str],
        cursor: Mapping[str, Any] | None,
        limit: int,
        max_pages: int,
        http_client: httpx.AsyncClient | None = None,
    ) -> SearchBatchResult:
        api_key = credentials.get("api_key", "")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Google Scholar requires an explicit api_key")
        arguments = source_arguments(
            query_args,
            settings,
            cursor,
            limit,
            max_pages,
            page_field="num",
            offset_field="start",
            page_max=20,
        )
        arguments = GoogleScholarSearchArgs.model_validate(arguments).model_dump()
        try:
            result = await google_scholar_search_service(
                arguments, http_client=http_client, api_key=api_key
            )
        except (httpx.HTTPError, ToolException) as exc:
            message = str(exc)
            for value in credentials.values():
                if value:
                    message = message.replace(value, "[redacted]")
            return failed_result(self.provider_id, message)
        for value in credentials.values():
            if value:
                result["warnings"] = [
                    warning.replace(value, "[redacted]")
                    for warning in result["warnings"]
                ]
        return batch_result(self.provider_id, result)

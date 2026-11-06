from collections.abc import Mapping
from typing import Any

import httpx
from langchain_core.tools import ToolException

from app.llm.tools.search_tools.dblp.search_dblp import (
    DblpSearchArgs,
    dblp_search_service,
)
from app.utils.search_adapters.contracts import (
    SearchBatchResult,
    batch_result,
    failed_result,
    source_arguments,
)


class DblpAdapter:
    __slots__ = ()
    provider_id = "dblp"

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
        if credentials:
            raise ValueError("This provider does not accept credentials")
        arguments = source_arguments(
            query_args,
            settings,
            cursor,
            limit,
            max_pages,
            page_field="h",
            offset_field="f",
            page_max=1000,
        )
        arguments = DblpSearchArgs.model_validate(arguments).model_dump()
        try:
            result = await dblp_search_service(arguments, http_client=http_client)
        except (httpx.HTTPError, ToolException) as exc:
            message = str(exc)
            return failed_result(self.provider_id, message)
        return batch_result(self.provider_id, result)

from collections.abc import Mapping
from datetime import date
import json
from math import ceil
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.errors import GraphBubbleUp

from app.llm.context import PaperSearchContext
from app.llm.graph.workflows.paper_search_schemas import SearchPlanningArgs, SearchTagArgs, SourceTypeValue
from app.llm.provider import ValidatedJsonInvoker
from app.services.search_connection_executor import search_executor as shared_search_executor, validate_query
from app.utils.search_adapters.contracts import SEARCH_PAGE_SIZE


PLAN_SEARCH_SYSTEM_PROMPT = """
你是论文检索规划专家。根据用户原始请求及 available_providers 生成 SearchPlanningArgs JSON。
query_understanding.topic、keywords、synonyms、subfields、includeTerms、excludeTerms 使用英文学术表达，reasoning 使用中文。
只保留用户明确的限制；只有明确给出起止年份时填写 yearFrom/yearTo，相对年份不推断。
requiresCode 未明确要求时为 null；intent 只能 survey、benchmark、method、mixed。
paperTag 未限定时使用 [1, 2]。query_plans 必须恰好覆盖 available_providers，不添加其他来源。
每个计划包含 display_query、reasoning、arguments；arguments 不得包含任何分页字段。
arxiv 支持 query、search_type(topic/title/author/abstract/category)、category、year_from/year_to、sort(relevance/newest/updated)、include_abstract；年份不早于1991。
dblp 支持 query、year_from/year_to、venue、paper_type；类型未限定时使用 all。
google_scholar 支持 query、year_from/year_to、review_only。
不能直接表达的限制由后续过滤处理，不伪造 API 参数。
""".strip()

_PROVIDER_TAGS = {
    "arxiv": SourceTypeValue.ARXIV,
    "dblp": SourceTypeValue.DBLP,
    "google_scholar": SourceTypeValue.GOOGLE_SCHOLAR,
}


class PlanSearchNode:
    def __init__(self, *, model: BaseChatModel, search_executor=None):
        self.executor = search_executor if search_executor is not None else shared_search_executor
        self.model = model
        self.invoker = ValidatedJsonInvoker(model=model, schema=SearchPlanningArgs)

    async def __call__(self, state: Mapping[str, Any], runtime: Runtime[PaperSearchContext]) -> dict:
        try:
            available, warnings = await self.executor.get_available_providers(
                user_id=runtime.context["user_id"], provider_ids=state.get("provider_ids"),
            )
            output = await self.invoker.ainvoke([
                SystemMessage(content=PLAN_SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({
                    "prompt": state["original_prompt"], "available_providers": list(available),
                }, ensure_ascii=False)),
            ])
            output = SearchPlanningArgs.model_validate(output)
            if set(output.query_plans) != set(available):
                raise ValueError("query_plans must exactly match available providers")
            understanding = output.query_understanding
            tasks = {}
            for provider, connection in available.items():
                arguments = dict(output.query_plans[provider].arguments)
                # Explicit intent, rather than model-invented dates, determines the range.
                arguments.pop("year_from", None)
                arguments.pop("year_to", None)
                if understanding.yearFrom is not None:
                    arguments.update(year_from=understanding.yearFrom, year_to=understanding.yearTo)
                validate_query(provider, arguments)
                initial_limit = connection["initial_limit"]
                tasks[provider] = {
                    "provider_id": provider, "round": 0, "query_args": arguments,
                    "configuration_version": connection["configuration_version"],
                    "cursor": None, "limit": initial_limit,
                    "max_pages": ceil(initial_limit / SEARCH_PAGE_SIZE),
                }
            year_tag = (date.today().year - understanding.yearFrom + 1
                        if understanding.yearTo == date.today().year else 0)
            tags = SearchTagArgs(yearTag=year_tag, paperTag=output.paperTag,
                                 sourceTag=[_PROVIDER_TAGS[p] for p in tasks])
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {"stage": "failed", "status": "failed", "error": f"论文检索计划失败：{exc}"}
        return {
            "stage": "searching", "status": "running", "error": None,
            "query_understanding": understanding.model_dump(mode="json"),
            "search_tag": tags.model_dump(mode="json"),
            "provider_query_plans": tasks, "active_search_tasks": tasks,
            "supplemental_search_round": 0,
            "warnings": [*state.get("warnings", []), *warnings],
        }

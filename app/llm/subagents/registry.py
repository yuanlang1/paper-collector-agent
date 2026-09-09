from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    requires_confirmation: bool = True
    display_name: str | None = None
    confirmation_summary: str | None = None

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


@dataclass(frozen=True)
class SubAgentStreamSpec:
    workflow: str


@dataclass(frozen=True)
class SubAgentRuntime:
    spec: SubAgentSpec
    graph: Any
    error_code: str
    failure_summary: str
    stream: SubAgentStreamSpec


class SubAgentRegistry:
    def __init__(self, runtimes: Sequence[SubAgentRuntime]):
        self._runtimes = {runtime.spec.name: runtime for runtime in runtimes}
        self._by_stream_workflow = {
            runtime.stream.workflow: runtime
            for runtime in runtimes
        }
        if len(self._runtimes) != len(runtimes):
            raise ValueError("子代理名称不能重复。")
        if len(self._by_stream_workflow) != len(runtimes):
            raise ValueError("子代理流式工作流标识不能重复。")

        for runtime in runtimes:
            if runtime.graph is None:
                raise ValueError(f"子代理 {runtime.spec.name} 缺少执行图。")
            if not runtime.error_code or not runtime.failure_summary:
                raise ValueError(f"子代理 {runtime.spec.name} 缺少失败策略。")

    def all_specs(self) -> tuple[SubAgentSpec, ...]:
        return tuple(runtime.spec for runtime in self._runtimes.values())

    def all(self) -> tuple[SubAgentSpec, ...]:
        return self.all_specs()

    def get_spec(self, name: str | None) -> SubAgentSpec | None:
        runtime = self.get_runtime(name)
        return runtime.spec if runtime else None

    def get(self, name: str | None) -> SubAgentSpec | None:
        return self.get_spec(name)

    def get_runtime(self, name: str | None) -> SubAgentRuntime | None:
        return self._runtimes.get(name) if name else None

    def get_by_stream_workflow(
        self,
        workflow: str | None,
    ) -> SubAgentRuntime | None:
        return self._by_stream_workflow.get(workflow) if workflow else None

    def descriptions(self) -> str:
        return "\n".join(
            f"- {spec.name}: {spec.description}"
            for spec in self.all_specs()
        )


def build_default_subagent_registry(
    *,
    source_query_plan_model: Any,
) -> SubAgentRegistry:
    from app.llm.subagents.paper_search import build_paper_search_runtime
    from app.llm.subagents.task_indexing import build_task_indexing_runtime
    from app.llm.subagents.task_review import build_task_review_runtime

    return SubAgentRegistry(
        (
            build_paper_search_runtime(
                source_query_plan_model=source_query_plan_model,
            ),
            build_task_indexing_runtime(),
            build_task_review_runtime(),
        )
    )

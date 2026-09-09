from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.llm.streaming.notify import langgraph_notifier


Node = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class TimelineStep:
    key: str
    label: str
    starts_at: frozenset[str]
    completes_at: frozenset[str]
    nodes: frozenset[str]
    repeats: bool = False


PAPER_SEARCH_TIMELINE = (
    TimelineStep(
        key="prepare",
        label="准备检索",
        starts_at=frozenset({"initialize"}),
        completes_at=frozenset({"build_search_tag", "confirm"}),
        nodes=frozenset({
            "initialize",
            "intent_understanding",
            "build_search_tag",
            "confirm",
        }),
    ),
    TimelineStep(
        key="query_plan",
        label="生成检索计划",
        starts_at=frozenset({"generate_queries"}),
        completes_at=frozenset({"generate_queries"}),
        nodes=frozenset({"generate_queries"}),
    ),
    TimelineStep(
        key="multi_source_search",
        label="多来源检索",
        starts_at=frozenset({"search_arxiv"}),
        completes_at=frozenset({"finalize_source_search"}),
        nodes=frozenset({
            "search_arxiv",
            "search_dblp",
            "search_google",
            "finalize_source_search",
        }),
        repeats=True,
    ),
    TimelineStep(
        key="quality_review",
        label="清洗与检索复盘",
        starts_at=frozenset({"filter"}),
        completes_at=frozenset({"search_review"}),
        nodes=frozenset({"filter", "search_review"}),
        repeats=True,
    ),
    TimelineStep(
        key="supplemental_search_plan",
        label="补充检索计划",
        starts_at=frozenset({"supplemental_search"}),
        completes_at=frozenset({"supplemental_search"}),
        nodes=frozenset({"supplemental_search"}),
        repeats=True,
    ),
    TimelineStep(
        key="enrichment",
        label="补充论文信息",
        starts_at=frozenset({"enrich"}),
        completes_at=frozenset({"abstract_enrich"}),
        nodes=frozenset({
            "enrich",
            "crossref_enrich",
            "venue",
            "download_pdf",
            "abstract_enrich",
        }),
    ),
    TimelineStep(
        key="recommendation",
        label="论文推荐",
        starts_at=frozenset({"recommend"}),
        completes_at=frozenset({"recommend"}),
        nodes=frozenset({"recommend"}),
    ),
    TimelineStep(
        key="persist_and_sync",
        label="保存结果与同步任务状态",
        starts_at=frozenset({"create_task"}),
        completes_at=frozenset({"finalize_result"}),
        nodes=frozenset({
            "create_task",
            "persist",
            "cleanup_downloaded_pdfs",
            "update_task_status",
            "finalize_result",
        }),
    ),
)


TASK_REVIEW_TIMELINE = (
    TimelineStep(
        key="load_corpus",
        label="加载任务语料",
        starts_at=frozenset({"load_task_corpus"}),
        completes_at=frozenset({"load_task_corpus"}),
        nodes=frozenset({"load_task_corpus"}),
    ),
    TimelineStep(
        key="framework",
        label="生成综述框架",
        starts_at=frozenset({"generate_framework"}),
        completes_at=frozenset({"generate_framework"}),
        nodes=frozenset({"generate_framework"}),
    ),
    TimelineStep(
        key="claims",
        label="生成或修订论点",
        starts_at=frozenset({"generate_claims"}),
        completes_at=frozenset({"generate_claims"}),
        nodes=frozenset({"generate_claims"}),
        repeats=True,
    ),
    TimelineStep(
        key="evidence",
        label="检索或补充证据",
        starts_at=frozenset({"retrieve_evidence"}),
        completes_at=frozenset({"retrieve_evidence"}),
        nodes=frozenset({"retrieve_evidence"}),
        repeats=True,
    ),
    TimelineStep(
        key="render_sections",
        label="撰写或重写章节",
        starts_at=frozenset({"render_sections"}),
        completes_at=frozenset({"render_sections"}),
        nodes=frozenset({"render_sections"}),
        repeats=True,
    ),
    TimelineStep(
        key="assemble_review",
        label="组装综述",
        starts_at=frozenset({"assemble_review"}),
        completes_at=frozenset({"assemble_review"}),
        nodes=frozenset({"assemble_review"}),
        repeats=True,
    ),
    TimelineStep(
        key="reflection",
        label="反思与质量检查",
        starts_at=frozenset({"reflect_review"}),
        completes_at=frozenset({"reflect_review"}),
        nodes=frozenset({"reflect_review"}),
        repeats=True,
    ),
    TimelineStep(
        key="persist_review",
        label="保存综述",
        starts_at=frozenset({"finalizing_handoff"}),
        completes_at=frozenset({"finalize_result"}),
        nodes=frozenset({
            "finalizing_handoff",
            "persist_review",
            "finalize_result",
        }),
    ),
)


TASK_INDEXING_TIMELINE = (
    TimelineStep(
        key="prepare",
        label="校验任务状态",
        starts_at=frozenset({"initialize"}),
        completes_at=frozenset({"check_rag_status"}),
        nodes=frozenset({"initialize", "check_rag_status"}),
    ),
    TimelineStep(
        key="index",
        label="索引论文到知识库",
        starts_at=frozenset({"run_or_wait"}),
        completes_at=frozenset({"run_or_wait"}),
        nodes=frozenset({"run_or_wait"}),
    ),
    TimelineStep(
        key="verify",
        label="确认索引结果",
        starts_at=frozenset({"verify_completion"}),
        completes_at=frozenset({"verify_completion"}),
        nodes=frozenset({"verify_completion"}),
    ),
    TimelineStep(
        key="finalize",
        label="汇总索引结果",
        starts_at=frozenset({"finalize_result"}),
        completes_at=frozenset({"finalize_result"}),
        nodes=frozenset({"finalize_result"}),
    ),
)


def _step_for_node(
    timeline: tuple[TimelineStep, ...],
    node_name: str,
) -> TimelineStep | None:
    return next(
        (step for step in timeline if node_name in step.nodes),
        None,
    )


def _event_payload(
    *,
    workflow: str,
    step: TimelineStep,
    state: Mapping[str, Any],
    event_state: str,
    round_key: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    round_number = (
        int(state.get(round_key, 0)) + 1
        if round_key
        else None
    )
    iteration = round_number if step.repeats else None
    label = (
        f"{step.label}（第 {round_number} 轮）"
        if iteration is not None
        else step.label
    )
    suffix = f":{iteration}" if iteration is not None else ""

    return {
        "step_id": f"{workflow}:{step.key}{suffix}",
        "step_key": step.key,
        "label": label,
        "state": event_state,
        "iteration": iteration,
        "error": error,
    }


def instrument_timeline_node(
    *,
    workflow: str,
    node_name: str,
    node: Node,
    timeline: tuple[TimelineStep, ...],
    round_key: str | None = None,
) -> Node:
    step = _step_for_node(timeline, node_name)
    if step is None:
        return node

    accepts_config = "config" in inspect.signature(node).parameters

    async def wrapped(
        state: Mapping[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        notify = langgraph_notifier(config).scoped(
            workflow=workflow,
            node=node_name,
        )
        if node_name in step.starts_at:
            notify("timeline_step", _event_payload(
                workflow=workflow,
                step=step,
                state=state,
                event_state="started",
                round_key=round_key,
            ))

        try:
            result = await (
                node(state, config) if accepts_config else node(state)
            )
        except Exception as exc:
            notify("timeline_step", _event_payload(
                workflow=workflow,
                step=step,
                state=state,
                event_state="failed",
                round_key=round_key,
                error=str(exc),
            ))
            raise

        error = result.get("error")
        if error:
            notify("timeline_step", _event_payload(
                workflow=workflow,
                step=step,
                state=state,
                event_state="failed",
                round_key=round_key,
                error=str(error),
            ))
        elif node_name in step.completes_at:
            notify("timeline_step", _event_payload(
                workflow=workflow,
                step=step,
                state=state,
                event_state="completed",
                round_key=round_key,
            ))

        return result

    return wrapped

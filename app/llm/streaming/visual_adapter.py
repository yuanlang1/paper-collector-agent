from __future__ import annotations

from typing import Any

from app.llm.streaming.utils import (
    to_jsonable,
)


StreamEvent = tuple[
    str,
    dict[str, Any],
]

PAPER_SEARCH_PHASES = (
    ("prepare", "准备检索"),
    ("plan", "生成检索计划"),
    ("search", "多来源检索"),
    ("filter", "清洗与质量审核"),
    ("enrich", "论文信息补充"),
    ("recommend", "论文推荐"),
    ("persist", "保存与清理"),
    ("finalize", "同步任务状态"),
)

PAPER_SEARCH_NODE_PHASE = {
    "initialize": "prepare",
    "intent_understanding": "prepare",
    "build_search_tag": "prepare",
    "confirm": "prepare",
    "create_task": "persist",
    "generate_queries": "plan",
    "search_arxiv": "search",
    "search_dblp": "search",
    "search_google": "search",
    "finalize_source_search": "search",
    "filter": "filter",
    "search_review": "filter",
    "supplemental_search": "filter",
    "enrich": "enrich",
    "crossref_enrich": "enrich",
    "venue": "enrich",
    "download_pdf": "enrich",
    "abstract_enrich": "enrich",
    "recommend": "recommend",
    "persist": "persist",
    "cleanup_downloaded_pdfs": "persist",
    "update_task_status": "finalize",
    "finalize_handoff": "finalize",
}

TASK_REVIEW_PHASES = (
    ("prepare", "初始化与语料校验"),
    ("framework", "生成综述框架"),
    ("claims", "生成论点"),
    ("evidence", "检索证据"),
    ("render", "撰写章节"),
    ("review", "反思与修订"),
    ("finalize", "生成最终综述"),
)

TASK_REVIEW_NODE_PHASE = {
    "initialize": "prepare",
    "load_task_corpus": "prepare",
    "generate_framework": "framework",
    "generate_claims": "claims",
    "retrieve_evidence": "evidence",
    "render_sections": "render",
    "assemble_review": "render",
    "reflect_review": "review",
    "finalizing_handoff": "finalize",
    "finalize_task_review": "finalize",
}

class AgentStreamAdapter:
    def __init__(self):
        self._seen_action_results: set[str] = set()

    def handle_update(
        self,
        *,
        node_name: str,
        update: dict[str, Any],
    ) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        result = to_jsonable(update.get("last_action_result"))

        if isinstance(result, dict):
            action_id = result.get("action_id")

            if (action_id and str(action_id) not in self._seen_action_results):
                self._seen_action_results.add(str(action_id))

                events.append(
                    (
                        "action_result",
                        self._result_payload(
                            result=result,
                            node_name=node_name,
                        ),
                    )
                )

        run_status = self._string_value(update.get("run_status"))

        if (run_status == "failed" or update.get("error")):
            events.append(
                (
                    "node_error",
                    {
                        "node": node_name,
                        "error": update.get("error") or "节点执行失败"
                    },
                )
            )

        return events

    def handle_custom(
        self,
        payload: Any,
    ) -> list[StreamEvent]:
        normalized = to_jsonable(payload)

        if not isinstance(normalized, dict):
            return []

        event_name = normalized.get("event")

        if not event_name:
            return [(
                    "progress",
                    normalized,
                )
            ]

        data = {
            key: value
            for key, value in normalized.items()
            if key != "event"
        }

        return [(
                str(event_name),
                data,
            )
        ]

    def confirmation_required(
        self,
        payload: Any,
    ) -> StreamEvent:
        return (
            "confirmation_required",
            {
                "interrupt": to_jsonable(payload)
            },
        )

    @staticmethod
    def subagent_progress(
        *,
        subagent: str,
        action_id: str | None,
        child_thread_id: str,
        checkpoint_namespace: str,
        node_name: str,
        update: dict[str, Any],
        workflow: str = "paper_search",
    ) -> StreamEvent:
        phases, node_phases = (
            (TASK_REVIEW_PHASES, TASK_REVIEW_NODE_PHASE)
            if workflow == "task_review"
            else (PAPER_SEARCH_PHASES, PAPER_SEARCH_NODE_PHASE)
        )
        phase_key = node_phases.get(
            node_name,
            "prepare",
        )
        phase_index = next(
            (
                index
                for index, (key, _label) in enumerate(
                    phases,
                    start=1,
                )
                if key == phase_key
            ),
            1,
        )
        phase_label = dict(phases)[phase_key]
        stage = update.get("stage")
        terminal = node_name in {"finalize_handoff", "finalize_task_review"}
        progress_percent = (
            100
            if terminal
            else round(
                phase_index / len(phases) * 100
            )
        )

        return (
            "subagent_progress",
            {
                "subagent": subagent,
                "workflow": workflow,
                "delegation_id": action_id,
                "child_thread_id": child_thread_id,
                "checkpoint_namespace": checkpoint_namespace,
                "node": node_name,
                "phase": phase_key,
                "phase_label": phase_label,
                "phase_index": phase_index,
                "phase_count": len(phases),
                "progress_percent": progress_percent,
                "terminal": terminal,
                "task_id": update.get("paper_service_task_id") or update.get("task_id"),
                "stage": stage,
                "status": update.get("status"),
                "progress": to_jsonable(update.get("progress") or {}),
                "warnings": to_jsonable(update.get("warnings") or []),
                "error": update.get("error"),
                "task_status_update_error": update.get(
                    "task_status_update_error"
                ),
                "pdf_cleanup_error": update.get(
                    "pdf_cleanup_error"
                ),
                "degraded": bool(update.get("degraded")),
                "remote_task_state": update.get(
                    "remote_task_state"
                ),
                "supplemental_search_round": update.get(
                    "supplemental_search_round",
                    0,
                ),
                "iteration": (
                    int(update.get("reflection_round", 0)) + 1
                    if workflow == "task_review"
                    else int(update.get("supplemental_search_round", 0)) + 1
                ),
                "source_stats": to_jsonable(
                    update.get("source_search_stats") or {}
                ),
            },
        )

    @staticmethod
    def _result_payload(
        *,
        result: dict[str, Any],
        node_name: str,
    ) -> dict[str, Any]:
        artifact_refs = result.get("artifact_refs")

        return {
            "node": node_name,
            "action_id": result.get("action_id"),
            "action_type": result.get("action_type"),
            "name": result.get("name"),
            "status": result.get("status"),
            "summary": result.get("summary"),
            "artifact_refs": (
                artifact_refs
                if isinstance(artifact_refs, list)
                else []
            ),
            "retryable": bool(result.get("retryable")),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        }

    @staticmethod
    def _string_value(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if hasattr(value, "value"):
            return str(value.value)

        return str(value)

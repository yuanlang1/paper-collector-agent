from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.provider import ValidatedJsonInvoker
from app.llm.tools.venue_tools.easy_scholar import easy_scholar_venue_handler
from app.schemas.venue import VenueInfoDraft, VenueRecord
from app.services.venue_service import VenueService

VENUE_FALLBACK_SYSTEM_PROMPT = """
你是学术 venue 信息规范化员，负责根据论文标题和来源提供的候选名称生成标准期刊或会议记录。

规则：
- standard_name 必须是可确认的标准 venue 名称。
- 优先填写可靠的常用 acronym；无法确认时使用 standard_name，不得编造缩写。
- type 无法确认时为 0。
- 不得编造 SCI、CCF、影响因子、中科院分区或核心等级。
- 没有可靠依据时，sci_rank、ccf_rank、sci_up、sci_up_small、core_rank 必须为 null，sci_if 必须为 0。

输出规范：
- 只输出合法 JSON 对象，不要输出 Markdown、代码块或额外说明。
- 必须包含全部 venue 字段。

输出样例：
{
  "standard_name": "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
  "acronym": "ACL",
  "type": 2,
  "sci_rank": null,
  "ccf_rank": null,
  "sci_if": 0,
  "sci_up": null,
  "sci_up_small": null,
  "core_rank": null
}
""".strip()


class ModelVenueInfoDraft(VenueInfoDraft):
    @model_validator(mode="after")
    def fill_empty_acronym(self) -> ModelVenueInfoDraft:
        if _text(self.acronym) is None:
            self.acronym = self.standard_name
        return self


def _text(value: Any) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def _decimal(value: Any) -> Decimal:
    normalized = _text(value)
    if normalized is None:
        return Decimal(0)

    try:
        result = Decimal(normalized)
    except InvalidOperation:
        return Decimal(0)

    return result if result.is_finite() else Decimal(0)


def _custom_rank_value(
    custom_ranks: Any,
    dataset_name: str,
) -> str | None:
    if not isinstance(custom_ranks, list):
        return None

    for custom_rank in custom_ranks:
        if not isinstance(custom_rank, Mapping):
            continue

        abb_name = _text(custom_rank.get("abbName"))
        current_rank = _text(custom_rank.get("currentRank"))

        if (
            abb_name is not None
            and abb_name.casefold() == dataset_name.casefold()
            and current_rank is not None
        ):
            return current_rank

    return None


def _crossref_paper_type_code(paper: Mapping[str, Any]) -> int:
    metadata = paper.get("crossref_metadata")
    if not isinstance(metadata, Mapping):
        return 0

    value = metadata.get("paper_type_code")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _build_easy_scholar_venue(
    data: Mapping[str, Any],
) -> VenueInfoDraft:
    official_rank = data.get("official_rank")
    official_rank = official_rank if isinstance(official_rank, Mapping) else {}

    publication_name = _text(data.get("publication_name"))
    custom_ccf_rank = _custom_rank_value(
        data.get("custom_rank"),
        "CCF",
    )

    if not publication_name:
        raise ValueError("EasyScholar 未返回 publication_name。")

    return VenueInfoDraft(
        standard_name=publication_name,
        acronym=_text(data.get("abbreviation")),
        type=0,
        sci_rank=_text(official_rank.get("sci")),
        ccf_rank=custom_ccf_rank or _text(official_rank.get("ccf")),
        sci_if=_decimal(official_rank.get("sciif") or official_rank.get("sciIf")),
        sci_up=_text(official_rank.get("sciUp")),
        sci_up_small=_text(official_rank.get("sciUpSmall")),
        core_rank=_text(official_rank.get("pku")),
    )


class VenueResolutionNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: BaseChatModel,
        easy_scholar_handler: Any | None = None,
        venue_service: VenueService | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.venue_service = venue_service or VenueService()
        self.model: BaseChatModel = model
        self.invoker = ValidatedJsonInvoker(
            model=self.model,
            schema=ModelVenueInfoDraft,
        )
        self.easy_scholar_handler = easy_scholar_handler or easy_scholar_venue_handler

    async def _query_easy_scholar(
        self,
        publication_name: str,
    ) -> dict[str, Any] | None:
        result = await self.easy_scholar_handler(
            {"publication_name": publication_name},
            None,
        )

        if result.get("ok") is not True:
            return None

        data = result.get("data")
        return data if isinstance(data, dict) else None

    async def resolve_venue(
        self,
        *,
        title: str,
        venue_candidates: list[str],
    ) -> tuple[VenueInfoDraft | None, str]:
        candidates = [
            candidate.strip()
            for candidate in venue_candidates
            if isinstance(candidate, str) and candidate.strip()
        ]

        if not candidates:
            return None, "missing"

        for candidate in dict.fromkeys(candidates):
            record, source = await self.venue_service.get_by_name(candidate)
            if record is not None:
                return record, source

        for candidate in candidates:
            tool_data = await self._query_easy_scholar(candidate)

            if tool_data is None:
                continue

            try:
                venue = _build_easy_scholar_venue(tool_data)
            except ValueError:
                continue

            record, source = await self.venue_service.get_by_name(venue.standard_name)
            return (record, source) if record is not None else (venue, "easy_scholar")

        result = await self.invoker.ainvoke(
            [
                SystemMessage(
                    content=VENUE_FALLBACK_SYSTEM_PROMPT,
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "paper_title": title,
                            "venue_candidates": candidates,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )

        model_venue = ModelVenueInfoDraft.model_validate(
            result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        )
        venue = VenueInfoDraft.model_validate(model_venue)

        record, source = await self.venue_service.get_by_name(venue.standard_name)
        return (record, source) if record is not None else (venue, "model_fallback")

    async def _resolve_paper(self, paper: dict[str, Any]) -> tuple[str, str | None]:
        paper_info = paper["paper_info"]
        paper_info["venue_id"] = None
        title = paper_info["title"]
        try:
            venue, source = await self.resolve_venue(
                title=title,
                venue_candidates=paper.get("venue_candidates", []),
            )
            if venue is None:
                paper["venue_resolution"] = {
                    "status": "missing",
                    "source": source,
                    "venue_id": None,
                }
                return "missing", None
            paper_type_code = _crossref_paper_type_code(paper)
            if paper_type_code and not isinstance(venue, VenueRecord):
                venue = venue.model_copy(update={"type": paper_type_code})
            venue_id = venue.id if isinstance(venue, VenueRecord) else None
            paper_info["venue_id"] = venue_id
            paper["venue_resolution"] = {
                "status": "resolved",
                "source": source,
                "venue_id": venue_id,
                "venue_draft": venue.model_dump(mode="json"),
            }
            return "resolved", None
        except Exception as exc:
            paper["venue_resolution"] = {
                "status": "failed",
                "source": None,
                "venue_id": None,
            }
            return "failed", f"{title} 的 venue 解析失败：{exc}"

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]
            artifact_uri = state.get(
                "crossref_enrichment_manifest_artifact_ref"
            )

            manifest = await self.artifact_store.read_json_uri(artifact_uri)
            papers = manifest.get("papers", [])

            resolved_count = 0
            missing_count = 0
            failed_count = 0
            warnings: list[str] = []

            for paper in papers:
                status, warning = await self._resolve_paper(paper)
                resolved_count += status == "resolved"
                missing_count += status == "missing"
                failed_count += status == "failed"
                if warning:
                    warnings.append(warning)

            manifest["papers"] = papers
            manifest["step_key"] = "resolve_venues"

            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="resolve_venues",
                source="venue",
                payload=manifest,
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"venue 信息补充失败：{exc}",
            }

        return {
            "stage": "downloading_pdfs",
            "status": "running",
            "venue_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "venue_resolved": resolved_count,
                "venue_missing": missing_count,
                "venue_failed": failed_count,
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }

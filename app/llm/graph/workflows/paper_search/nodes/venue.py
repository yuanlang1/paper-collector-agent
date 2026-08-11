from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.model_factory import create_validated_structured_chat_model
from app.llm.tools.venue_tools.easy_scholar import (
    EASY_SCHOLAR_VENUE_TOOL,
)


VENUE_FALLBACK_SYSTEM_PROMPT = """
    你负责规范化学术期刊或会议的 venue 信息。

    输入包含论文标题和来源提供的 venue 候选名称。

    规则：
    - standard_name 必须是可确认的标准 venue 名称；
    - acronym 必须填写该 venue 的常用缩写，不得为 null 或空字符串；
    - type 无法确认时为 0；
    - 不得编造 SCI、CCF、影响因子、中科院分区或核心等级；
    - 没有可靠依据时，sci_rank、ccf_rank、sci_up、sci_up_small、
    core_rank 必须为 null，sci_if 必须为 0。
    - If a common acronym can be inferred from standard_name or the venue candidates,
      populate acronym with it; acronym must not be null or empty.
""".strip()


class VenueInfoDraft(BaseModel):
    standard_name: str = Field(min_length=1, max_length=500)
    acronym: str | None = Field(default=None, max_length=100)
    type: int = 0

    sci_rank: str | None = Field(default=None, max_length=50)
    ccf_rank: str | None = Field(default=None, max_length=50)
    sci_if: Decimal = Decimal("0")
    sci_up: str | None = Field(default=None, max_length=100)
    sci_up_small: str | None = Field(default=None, max_length=200)
    core_rank: str | None = Field(default=None, max_length=20)


class ModelVenueInfoDraft(VenueInfoDraft):
    """Normalize model fallback output before it enters the venue cache."""

    @model_validator(mode="after")
    def fill_empty_acronym(self) -> "ModelVenueInfoDraft":
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
        return Decimal("0")

    try:
        result = Decimal(normalized)
    except InvalidOperation:
        return Decimal("0")

    return result if result.is_finite() else Decimal("0")


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


def _cache_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _crossref_paper_type_code(paper: Mapping[str, Any]) -> int:
    metadata = paper.get("crossref_metadata")
    if not isinstance(metadata, Mapping):
        return 0

    value = metadata.get("paper_type_code")
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        else 0
    )


class VenueCacheStore:
    """缓存已解析的 venue 草稿，避免重复调用外部检索服务。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    def _read_sync(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

        venues = payload.get("venues") if isinstance(payload, dict) else None
        if not isinstance(venues, dict):
            return {}

        return {
            key: value
            for key, value in venues.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _write_sync(self, venues: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"version": 2, "venues": venues},
                file,
                ensure_ascii=False,
                indent=2,
            )

        tmp_path.replace(self.path)

    async def get(
        self,
        candidates: list[str],
    ) -> VenueInfoDraft | None:
        async with self._lock:
            venues = await asyncio.to_thread(self._read_sync)

        for candidate in candidates:
            cached = venues.get(_cache_key(candidate))
            if cached is None:
                continue

            draft = dict(cached.get("venue_draft", cached))
            draft["sci_if"] = _decimal(draft.get("sci_if"))
            try:
                return VenueInfoDraft.model_validate(draft)
            except Exception:
                continue

        return None

    async def put(
        self,
        venue: VenueInfoDraft,
        aliases: list[str],
    ) -> None:
        keys = {
            _cache_key(value)
            for value in [*aliases, venue.standard_name, venue.acronym or ""]
            if value.strip()
        }
        payload = venue.model_dump(mode="json")

        async with self._lock:
            venues = await asyncio.to_thread(self._read_sync)
            existing_id = next(
                (
                    value.get("venue_id")
                    for value in venues.values()
                    if isinstance(value, dict)
                    and value.get("venue_draft", value).get(
                        "standard_name"
                    ) == venue.standard_name
                    and isinstance(value.get("venue_id"), int)
                ),
                None,
            )
            entry = {
                "venue_draft": payload,
                "venue_id": existing_id,
            }
            venues.update({key: entry for key in keys})
            await asyncio.to_thread(self._write_sync, venues)

    async def get_venue_id(self, standard_name: str) -> int | None:
        normalized_name = _cache_key(standard_name)
        async with self._lock:
            venues = await asyncio.to_thread(self._read_sync)

        for key, value in venues.items():
            draft = value.get("venue_draft", value)
            venue_id = value.get("venue_id")
            if (
                (key == normalized_name or _cache_key(
                    str(draft.get("standard_name") or "")
                ) == normalized_name)
                and isinstance(venue_id, int)
                and not isinstance(venue_id, bool)
                and venue_id > 0
            ):
                return venue_id

        return None

    async def set_venue_id(
        self,
        venue: VenueInfoDraft,
        venue_id: int,
    ) -> None:
        if not isinstance(venue_id, int) or isinstance(venue_id, bool) or venue_id <= 0:
            raise ValueError("venue_id must be a positive integer")

        normalized_name = _cache_key(venue.standard_name)
        async with self._lock:
            venues = await asyncio.to_thread(self._read_sync)
            matching_keys = [
                key
                for key, value in venues.items()
                if _cache_key(str(
                    value.get("venue_draft", value).get(
                        "standard_name"
                    ) or ""
                )) == normalized_name
            ]
            if not matching_keys:
                matching_keys = [normalized_name]

            entry = {
                "venue_draft": venue.model_dump(mode="json"),
                "venue_id": venue_id,
            }
            venues.update({key: entry for key in matching_keys})
            await asyncio.to_thread(self._write_sync, venues)


def _build_easy_scholar_venue(
    data: Mapping[str, Any],
) -> VenueInfoDraft:
    official_rank = data.get("official_rank")
    official_rank = (
        official_rank
        if isinstance(official_rank, Mapping)
        else {}
    )

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
        sci_if=_decimal(
            official_rank.get("sciif")
            or official_rank.get("sciIf")
        ),
        sci_up=_text(official_rank.get("sciUp")),
        sci_up_small=_text(
            official_rank.get("sciUpSmall")
        ),
        core_rank=_text(official_rank.get("pku")),
    )


class VenueResolutionNode:
    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        model: Any | None = None,
        easy_scholar_handler: Any | None = None,
        cache_store: VenueCacheStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.cache_store = cache_store or VenueCacheStore(
            self.artifact_store.base_dir.parent / "venue_cache.json"
        )
        self.model = model or create_validated_structured_chat_model(
            ModelVenueInfoDraft,
            temperature=0,
        )
        self.easy_scholar_handler = easy_scholar_handler or EASY_SCHOLAR_VENUE_TOOL.handler
    
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

        cached_venue = await self.cache_store.get(candidates)
        if cached_venue is not None:
            return cached_venue, "local_cache"

        for candidate in candidates:
            tool_data = await self._query_easy_scholar(candidate)

            if tool_data is None:
                continue

            try:
                venue = _build_easy_scholar_venue(tool_data)
            except ValueError:
                continue

            await self.cache_store.put(venue, candidates)
            return venue, "easy_scholar"

        result = await self.model.ainvoke(
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
            result.model_dump(mode="json")
            if isinstance(result, BaseModel)
            else result
        )
        venue = VenueInfoDraft.model_validate(model_venue)
        await self.cache_store.put(venue, candidates)

        return venue, "model_fallback"

    async def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            artifact_uri = state.get(
                "crossref_enrichment_manifest_artifact_ref"
            ) or state.get("enrichment_manifest_artifact_ref")

            if not isinstance(run_id, str) or not run_id:
                raise ValueError("缺少有效 run_id。")

            if (
                not isinstance(artifact_uri, str)
                or not artifact_uri.startswith("artifact://")
            ):
                raise ValueError("缺少 enrichment manifest artifact。")

            base_dir = self.artifact_store.base_dir.resolve()
            manifest_path = (
                base_dir
                / artifact_uri.removeprefix("artifact://")
            ).resolve()

            try:
                manifest_path.relative_to(base_dir)
            except ValueError as exc:
                raise ValueError("venue manifest 超出 artifact 存储目录。") from exc

            def read_manifest() -> dict[str, Any]:
                with manifest_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers", [])

            if not isinstance(papers, list):
                raise ValueError("manifest papers 格式无效。")

            resolved_count = 0
            missing_count = 0
            failed_count = 0
            warnings: list[str] = []
            retained_papers: list[dict[str, Any]] = []

            for paper in papers:
                if not isinstance(paper, dict):
                    continue

                paper_info = paper.get("paper_info")

                if not isinstance(paper_info, dict):
                    failed_count += 1
                    warnings.append("存在缺少 paper_info 的论文条目。")
                    continue

                title = str(paper_info.get("title") or "")
                venue_candidates = paper.get(
                    "venue_candidates",
                    [],
                )

                if not isinstance(venue_candidates, list):
                    venue_candidates = []

                try:
                    venue, source = await self.resolve_venue(
                        title=title,
                        venue_candidates=venue_candidates,
                    )

                    paper_info["venue_id"] = None

                    if venue is None:
                        paper["venue_resolution"] = {
                            "status": "missing",
                            "source": source,
                            "venue_id": None,
                        }
                        missing_count += 1
                        retained_papers.append(paper)
                        continue

                    paper_type_code = _crossref_paper_type_code(paper)
                    if paper_type_code:
                        venue = venue.model_copy(
                            update={"type": paper_type_code}
                        )

                    paper["venue_resolution"] = {
                        "status": "resolved",
                        "source": source,
                        "venue_id": None,
                        "venue_draft": venue.model_dump(mode="json"),
                    }
                    resolved_count += 1
                    retained_papers.append(paper)

                except Exception as exc:
                    paper_info["venue_id"] = None
                    paper["venue_resolution"] = {
                        "status": "failed",
                        "source": None,
                        "venue_id": None,
                    }
                    failed_count += 1
                    retained_papers.append(paper)
                    warnings.append(f"{title} 的 venue 解析失败：{exc}")

            manifest["papers"] = retained_papers
            manifest["step_key"] = "resolve_venues"

            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="resolve_venues",
                source="venue",
                kind="paper_info_venue_manifest_json",
                payload=manifest,
                count=len(retained_papers),
                metadata={
                    "input_manifest": artifact_uri,
                    "resolved_count": resolved_count,
                    "missing_count": missing_count,
                    "failed_count": failed_count,
                },
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"venue 信息补充失败：{exc}",
            }

        return {
            "stage": "downloading_pdfs",
            "status": (
                "partial_failed"
                if failed_count
                else "running"
            ),
            "degraded": (
                bool(state.get("degraded"))
                or failed_count > 0
            ),
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

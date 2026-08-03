from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from app.infrastructure.grpc.paper_service_grpc_client import (
    paper_service_grpc_client,
)
from app.infrastructure.grpc.task_service_grpc_client import (
    task_service_grpc_client,
)
from app.infrastructure.grpc.venue_service_grpc_client import (
    venue_service_grpc_client,
)
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.paper_cache import (
    PaperCacheStore,
)
from app.llm.graph.workflows.paper_search.nodes.venue import (
    VenueCacheStore,
    VenueInfoDraft,
)


BatchSave = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]


def _is_valid_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_persistable_venue_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _add_error(outcome: dict[str, Any], message: str) -> None:
    current = outcome.get("error")
    outcome["error"] = f"{current}; {message}" if current else message


def _venue_key(venue: VenueInfoDraft) -> str:
    return " ".join(venue.standard_name.split()).casefold()


class PersistRecommendedPapersNode:
    """Persist venues, papers, caches, and task-paper relations in order."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        save_venues: BatchSave | None = None,
        save_papers: BatchSave | None = None,
        save_task_papers: BatchSave | None = None,
        venue_cache_store: VenueCacheStore | None = None,
        paper_cache_store: PaperCacheStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        cache_dir = self.artifact_store.base_dir.parent
        self.save_venues = (
            save_venues or venue_service_grpc_client.batch_save_venues
        )
        self.save_papers = (
            save_papers or paper_service_grpc_client.batch_save_papers
        )
        self.save_task_papers = (
            save_task_papers
            or task_service_grpc_client.batch_save_task_paper_relations
        )
        self.venue_cache_store = venue_cache_store or VenueCacheStore(
            cache_dir / "venue_cache.json"
        )
        self.paper_cache_store = paper_cache_store or PaperCacheStore(
            cache_dir / "paper_cache.json"
        )

    @staticmethod
    def _outcome(
        client_key: str,
        title: str,
        paper_id: int | None,
    ) -> dict[str, Any]:
        return {
            "client_key": client_key,
            "title": title,
            "venue_id": None,
            "venue_save_status": "not_started",
            "paper_id": paper_id,
            "paper_save_status": (
                "not_required" if _is_valid_id(paper_id) else "pending"
            ),
            "paper_cache_status": "not_started",
            "task_paper_save_status": "pending",
            "error": None,
        }

    @staticmethod
    def _validate_entry(
        paper: Any,
        task_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any], int | None] | None:
        if not isinstance(paper, dict):
            return None

        paper_info = paper.get("paper_info")
        relation_draft = paper.get("task_paper_relation_draft")
        paper_id = paper.get("paper_id")
        if not isinstance(paper_info, dict) or not isinstance(
            relation_draft, dict
        ):
            return None
        if paper_id is not None and not _is_valid_id(paper_id):
            return None
        if relation_draft.get("search_task_id") != task_id:
            return None
        if relation_draft.get("paper_id") != paper_id:
            return None
        stars = relation_draft.get("recommendation_stars")
        reason = relation_draft.get("recommendation_reason")
        if (
            not isinstance(stars, int)
            or isinstance(stars, bool)
            or not 3 <= stars <= 5
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return None
        return paper, paper_info, paper_id

    async def _resolve_venues(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[int, int, int]:
        venues_to_save: dict[str, VenueInfoDraft] = {}
        cache_hits = 0

        for entry in entries:
            if entry["paper_id"] is not None:
                continue

            paper = entry["paper"]
            paper_info = entry["paper_info"]
            resolution = paper.get("venue_resolution")
            if not isinstance(resolution, dict):
                paper_info["venue_id"] = 0
                paper["venue_resolution"] = {
                    "status": "missing",
                    "source": None,
                    "venue_id": 0,
                }
                entry["outcome"]["venue_id"] = 0
                entry["outcome"]["venue_save_status"] = "not_available"
                continue

            venue_id = resolution.get("venue_id")
            if _is_valid_id(venue_id):
                paper_info["venue_id"] = venue_id
                entry["outcome"]["venue_id"] = venue_id
                entry["outcome"]["venue_save_status"] = "not_required"
                continue

            draft_data = resolution.get("venue_draft")
            try:
                venue = VenueInfoDraft.model_validate(draft_data)
            except Exception:
                paper_info["venue_id"] = 0
                resolution["venue_id"] = 0
                entry["outcome"]["venue_id"] = 0
                entry["outcome"]["venue_save_status"] = "not_available"
                continue

            cached_id = await self.venue_cache_store.get_venue_id(
                venue.standard_name
            )
            if cached_id is not None:
                paper_info["venue_id"] = cached_id
                resolution["venue_id"] = cached_id
                entry["outcome"]["venue_id"] = cached_id
                entry["outcome"]["venue_save_status"] = "cache_hit"
                cache_hits += 1
                continue

            entry["venue"] = venue
            venues_to_save.setdefault(_venue_key(venue), venue)

        if not venues_to_save:
            return cache_hits, 0, 0

        response = await self.save_venues(
            [venue.model_dump(mode="json") for venue in venues_to_save.values()]
        )
        saved = (
            (response.get("result") or {}).get("venues", [])
            if response.get("ok") is True
            else []
        )
        saved_ids = {
            " ".join(str(item.get("standard_name") or "").split()).casefold(): item.get("id")
            for item in saved
            if isinstance(item, dict) and _is_valid_id(item.get("id"))
        }
        error = str(response.get("error") or "batch venue save failed")
        saved_count = 0
        failed_count = 0

        for entry in entries:
            venue = entry.get("venue")
            if not isinstance(venue, VenueInfoDraft):
                continue
            venue_id = saved_ids.get(_venue_key(venue))
            if not _is_valid_id(venue_id):
                entry["outcome"]["venue_save_status"] = "failed"
                _add_error(entry["outcome"], error)
                failed_count += 1
                continue

            entry["paper_info"]["venue_id"] = venue_id
            entry["paper"]["venue_resolution"]["venue_id"] = venue_id
            entry["outcome"]["venue_id"] = venue_id
            entry["outcome"]["venue_save_status"] = "saved"
            await self.venue_cache_store.set_venue_id(venue, venue_id)
            saved_count += 1

        return cache_hits, saved_count, failed_count

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            run_id = state.get("run_id")
            task_id = state.get("paper_service_task_id")
            artifact_uri = state.get("recommendation_manifest_artifact_ref")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("missing valid run_id")
            if not _is_valid_id(task_id):
                raise ValueError("missing valid paper_service_task_id")
            if not isinstance(artifact_uri, str) or not artifact_uri.startswith(
                "artifact://"
            ):
                raise ValueError("missing recommendation manifest artifact")

            base_dir = self.artifact_store.base_dir.resolve()
            manifest_path = (
                base_dir / artifact_uri.removeprefix("artifact://")
            ).resolve()
            manifest_path.relative_to(base_dir)

            def read_manifest() -> dict[str, Any]:
                with manifest_path.open("r", encoding="utf-8") as file:
                    return json.load(file)

            manifest = await asyncio.to_thread(read_manifest)
            papers = manifest.get("papers")
            if not isinstance(papers, list):
                raise ValueError("recommendation manifest papers must be a list")

            outcomes: list[dict[str, Any]] = []
            entries: list[dict[str, Any]] = []
            invalid_count = 0
            for index, candidate in enumerate(papers):
                client_key = f"{run_id}:{index}"
                validated = self._validate_entry(candidate, task_id)
                title = (
                    str((candidate.get("paper_info") or {}).get("title") or "")
                    if isinstance(candidate, dict)
                    else ""
                )
                paper_id = candidate.get("paper_id") if isinstance(candidate, dict) else None
                outcome = self._outcome(client_key, title, paper_id)
                outcomes.append(outcome)
                if validated is None:
                    _add_error(outcome, "invalid recommended paper input")
                    outcome["paper_save_status"] = "not_started"
                    outcome["task_paper_save_status"] = "not_started"
                    invalid_count += 1
                    continue

                paper, paper_info, paper_id = validated
                entries.append(
                    {
                        "client_key": client_key,
                        "paper": paper,
                        "paper_info": paper_info,
                        "paper_id": paper_id,
                        "relation": {
                            "task_id": task_id,
                            "paper_id": paper_id,
                            "recommendation_stars": paper[
                                "task_paper_relation_draft"
                            ]["recommendation_stars"],
                            "recommendation_reason": paper[
                                "task_paper_relation_draft"
                            ]["recommendation_reason"].strip(),
                        },
                        "outcome": outcome,
                    }
                )

            venue_cache_hits, venue_saved, venue_failed = await self._resolve_venues(
                entries
            )
            new_requests = []
            for entry in entries:
                if entry["paper_id"] is not None:
                    continue
                if not _is_persistable_venue_id(
                    entry["paper_info"].get("venue_id")
                ):
                    entry["outcome"]["paper_save_status"] = "not_started"
                    entry["outcome"]["task_paper_save_status"] = "not_started"
                    continue
                info = entry["paper_info"]
                info["published_date"] = info.get("published_date") or info.get(
                    "publish_date"
                )
                new_requests.append(
                    {"client_key": entry["client_key"], "paper_info": info}
                )

            paper_response = await self.save_papers(new_requests) if new_requests else {"ok": True, "result": {"papers": []}}
            saved_papers = (
                (paper_response.get("result") or {}).get("papers", [])
                if paper_response.get("ok") is True
                else []
            )
            paper_ids = {
                item.get("client_key"): item.get("paper_id")
                for item in saved_papers
                if isinstance(item, dict) and _is_valid_id(item.get("paper_id"))
            }
            paper_error = str(paper_response.get("error") or "batch paper save failed")
            cache_updated = 0
            cache_failed = 0
            for entry in entries:
                if entry["paper_id"] is None:
                    paper_id = paper_ids.get(entry["client_key"])
                    if not _is_valid_id(paper_id):
                        if entry["outcome"]["paper_save_status"] != "not_started":
                            entry["outcome"]["paper_save_status"] = "failed"
                            _add_error(entry["outcome"], paper_error)
                        continue
                    entry["paper_id"] = paper_id
                    entry["paper"]["paper_id"] = paper_id
                    entry["relation"]["paper_id"] = paper_id
                    entry["outcome"]["paper_id"] = paper_id
                    entry["outcome"]["paper_save_status"] = "saved"

            for entry in entries:
                if not _is_valid_id(entry["paper_id"]):
                    continue
                try:
                    paper_for_cache = dict(entry["paper"])
                    paper_for_cache["paper_id"] = entry["paper_id"]
                    pdf_download = paper_for_cache.get("pdf_download")
                    if isinstance(pdf_download, dict):
                        paper_for_cache["pdf_download"] = {
                            key: value
                            for key, value in pdf_download.items()
                            if key not in {"local_pdf_path", "artifact_uri"}
                        }
                    await self.paper_cache_store.upsert_saved_paper(
                        paper_for_cache
                    )
                    entry["outcome"]["paper_cache_status"] = "saved"
                    cache_updated += 1
                except Exception as exc:
                    entry["outcome"]["paper_cache_status"] = "failed"
                    _add_error(entry["outcome"], f"paper cache failed: {exc}")
                    cache_failed += 1

            relation_requests = [
                entry["relation"]
                for entry in entries
                if _is_valid_id(entry["paper_id"])
            ]
            relation_response = (
                await self.save_task_papers(relation_requests)
                if relation_requests
                else {"ok": True, "result": {"saved_count": 0}}
            )
            saved_count = int(
                (relation_response.get("result") or {}).get("saved_count") or 0
            ) if relation_response.get("ok") is True else 0
            relation_error = str(
                relation_response.get("error") or "batch task-paper save failed"
            )
            relation_status = (
                "saved"
                if saved_count == len(relation_requests)
                else "partial_unknown"
                if relation_response.get("ok") is True
                else "failed"
            )
            for entry in entries:
                if _is_valid_id(entry["paper_id"]):
                    entry["outcome"]["task_paper_save_status"] = relation_status
                    if relation_status != "saved":
                        _add_error(entry["outcome"], relation_error)

            persisted_count = sum(
                outcome["task_paper_save_status"] == "saved"
                for outcome in outcomes
            )
            failure_count = len(outcomes) - persisted_count
            artifact = await self.artifact_store.write_json(
                run_id=run_id,
                step_key="persist_recommended_papers",
                source="paper_service",
                kind="persisted_papers_manifest_json",
                payload={
                    "run_id": run_id,
                    "step_key": "persist_recommended_papers",
                    "input_manifest": artifact_uri,
                    "results": outcomes,
                },
                count=persisted_count,
                metadata={
                    "input_count": len(papers),
                    "invalid_input_count": invalid_count,
                    "venue_cache_hit_count": venue_cache_hits,
                    "venue_saved_count": venue_saved,
                    "venue_failed_count": venue_failed,
                    "paper_saved_count": sum(
                        outcome["paper_save_status"] == "saved"
                        for outcome in outcomes
                    ),
                    "paper_cache_updated_count": cache_updated,
                    "paper_cache_failed_count": cache_failed,
                    "relation_saved_count": persisted_count,
                },
            )
        except Exception as exc:
            return {"stage": "failed", "status": "failed", "error": str(exc)}

        degraded = (
            bool(state.get("degraded"))
            or failure_count > 0
            or cache_failed > 0
        )
        if failure_count and not persisted_count:
            stage = "failed"
        elif degraded:
            stage = "partial_failed"
        else:
            stage = "completed"
        warnings = list(state.get("warnings", []))
        if failure_count:
            warnings.append(
                f"{failure_count} papers were not fully persisted"
            )
        if cache_failed:
            warnings.append(
                f"{cache_failed} papers were not written to paper cache"
            )
        return {
            "stage": stage,
            "status": stage,
            "degraded": degraded,
            "persisted_papers_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "persistence_input": len(papers),
                "persistence_saved": persisted_count,
                "persistence_failed": failure_count,
            },
            "warnings": warnings,
            "error": (
                "partial persistence failure"
                if failure_count
                else "paper cache update failure"
                if cache_failed
                else state.get("error")
            ),
        }

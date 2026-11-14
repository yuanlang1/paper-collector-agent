from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from anyio import fail_after
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

from app.config import settings
from app.infrastructure.http import http_manager
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.context import PaperSearchContext
from app.llm.graph.workflows.paper_search.nodes.venue import VenueResolver
from app.llm.graph.workflows.paper_search.paper_search_schemas import (
    PaperFrontPageMetadata,
)
from app.llm.graph.workflows.paper_search.state import EnrichWorkerState
from app.llm.prompt import FRONT_PAGE_PROMPT
from app.llm.provider import ChatClient, ModelOptions
from app.rag.file_parser.file_parser import FileParser
from app.schemas.file_parser import FileParseOptions, FileParseRequest, FileType
from app.services.llm_setting_service import get_small_llm_config
from app.tasks.modules.paper_search.crossref_enrich import normalize_doi


class PaperEnrichmentNode:
    def __init__(
        self,
        *,
        file_parser: FileParser | None = None,
        venue_resolver: VenueResolver | None = None,
    ) -> None:
        self.file_parser = file_parser
        self.venue_resolver = venue_resolver or VenueResolver()

    async def __call__(
        self,
        state: EnrichWorkerState,
        runtime: Runtime[PaperSearchContext],
    ) -> dict[str, Any]:
        index = state["paper_index"]
        info = dict(state["paper"]["paper_info"])
        paper = {**state["paper"], "paper_info": info}
        title = info.get("title") or f"候选论文 {index + 1}"
        warnings = []
        venue_name = None
        extraction_failed = False

        try:
            with fail_after(settings.PAPER_ABSTRACT_TASK_TIMEOUT_SECONDS):
                parser = self.file_parser or FileParser(
                    mineru_client=http_manager.mineru_client
                )
                file_id = f"{state['child_run_id']}-{index}"
                document = await parser.parse_one(
                    FileParseRequest(
                        file_id=file_id,
                        uri=info["pdf_url"],
                        file_name=f"{file_id}.pdf",
                        file_type=FileType.PDF,
                        options=FileParseOptions(page_ranges="1-2"),
                    )
                )
                if not document.page_content.strip():
                    raise ValueError("MinerU 返回空 Markdown")

                config = await get_small_llm_config(
                    user_id=int(runtime.context["user_id"])
                )
                invoker = ChatClient(config).structured(
                    PaperFrontPageMetadata,
                    options=ModelOptions(streaming=False),
                )
                metadata = await invoker.ainvoke(
                    [
                        SystemMessage(content=FRONT_PAGE_PROMPT),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "venue_candidates": paper.get("venue_candidates", []),
                                    "pdf_front_pages": document.page_content[:32_000],
                                },
                                ensure_ascii=False,
                            )
                        ),
                    ]
                )
        except GraphBubbleUp:
            raise
        except Exception as exc:
            extraction_failed = True
            warnings.append(
                f"{title} 的前两页提取失败（{type(exc).__name__}），保留来源信息。"
            )
        else:
            doi = normalize_doi(metadata.doi)
            source_doi = normalize_doi(info.get("doi"))
            abstract = (metadata.abstract or "").strip()
            if doi and source_doi and doi.casefold() != source_doi.casefold():
                warnings.append(f"{title} 的 PDF DOI 与来源 DOI 不一致，保留来源 DOI。")
                if abstract:
                    info["paper_abstract"] = abstract
                    info["abstract_source"] = "pdf_front_pages"
            else:
                if not info.get("title") and metadata.title:
                    info["title"] = metadata.title.strip()
                if not info.get("authors"):
                    authors = [
                        name.strip() for name in metadata.authors if name.strip()
                    ]
                    author_text = ", ".join(authors)
                    if 0 < len(author_text) <= 1000:
                        info["authors"] = author_text
                        paper["author_names"] = authors
                if doi and not source_doi:
                    info["doi"] = doi
                if abstract and (
                    not info.get("paper_abstract")
                    or info.get("abstract_source") == "search_snippet"
                ):
                    info["paper_abstract"] = abstract
                    info["abstract_source"] = "pdf_front_pages"
                keywords = [word.strip() for word in metadata.keywords if word.strip()]
                if keywords:
                    info["keywords"] = keywords
                ai_abstract = (metadata.ai_abstract or "").strip()
                if ai_abstract:
                    info["ai_abstract"] = ai_abstract
            venue_name = (metadata.venue_name or "").strip() or None

        eligible = bool(info.get("title") and info.get("authors"))
        if eligible:
            _, venue_warning = await self.venue_resolver.resolve_paper(
                paper, venue_name, use_source_fallback=extraction_failed
            )
            if venue_warning:
                warnings.append(venue_warning)
        else:
            warnings.append(f"{title} 缺少入库所需的标题或作者，已跳过。")
        return {
            "enrich_results": {
                index: {
                    "paper": paper if eligible else None,
                    "warnings": warnings,
                    "extraction_failed": extraction_failed,
                },
            },
        }


class CollectEnrichNode:
    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            results = state.get("enrich_results", {})
            ordered = [
                results[index] for index in range(len(state["enrich_candidates"]))
            ]
            papers = [item["paper"] for item in ordered if item["paper"] is not None]
            warnings = [
                *state.get("warnings", []),
                *(warning for item in ordered for warning in item["warnings"]),
            ]
            progress = {
                **state.get("progress", {}),
                "enrich_kept": len(papers),
                "enrich_extraction_failed": sum(
                    item["extraction_failed"] for item in ordered
                ),
            }
            if not papers and not progress.get("existing_in_database"):
                return {
                    "stage": "failed",
                    "status": "failed",
                    "progress": progress,
                    "warnings": warnings,
                    "error": "没有可继续推荐的论文。",
                }
            manifest = await self.artifact_store.read_json_uri(
                state["normalized_manifest_artifact_ref"]
            )
            artifact = await self.artifact_store.write_json(
                run_id=state["child_run_id"],
                step_key="enrich_papers",
                source="enrich",
                payload={**manifest, "step_key": "enrich_papers", "papers": papers},
            )
        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"论文补全结果汇总失败：{exc}",
            }
        return {
            "stage": "recommending",
            "enriched_manifest_artifact_ref": artifact.artifact_uri,
            "progress": progress,
            "warnings": warnings,
        }

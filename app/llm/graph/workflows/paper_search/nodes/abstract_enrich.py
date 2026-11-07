from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.context import PaperSearchContext
from app.tasks.modules.paper_search import abstract_enrich as abstract_tasks


class AbstractEnrichNode:
    def __init__(self, artifact_store=None):
        self.artifact_store = artifact_store or LocalArtifactStore()

    async def __call__(
        self,
        state: Mapping[str, Any],
        runtime: Runtime[PaperSearchContext],
    ) -> dict[str, Any]:
        try:
            child_run_id = state["child_run_id"]
            artifact_uri = state.get("pdf_manifest_artifact_ref")

            manifest = await self.artifact_store.read_json_uri(artifact_uri)
            papers = manifest.get("papers", [])

            result = await abstract_tasks.enrich_abstracts(
                papers, user_id=runtime.context["user_id"], child_run_id=child_run_id,
            )
            counts, warnings = result["counts"], result["warnings"]

            manifest["papers"] = papers
            manifest["step_key"] = "abstract_enrichment"

            artifact = await self.artifact_store.write_json(
                run_id=child_run_id,
                step_key="abstract_enrichment",
                source="abstract",
                payload=manifest,
            )

        except Exception as exc:
            return {
                "stage": "failed",
                "status": "failed",
                "error": f"摘要补充失败：{exc}",
            }

        return {
            "stage": "recommending",
            "status": "running",
            "abstract_manifest_artifact_ref": artifact.artifact_uri,
            "progress": {
                **state.get("progress", {}),
                "abstract_from_source": counts["source_abstract"],
                "abstract_extracted_from_pdf": (counts["extracted_abstract"]),
                "abstract_missing": counts["missing_abstract"],
                "pdf_front_pages_read": counts["front_pages_read"],
                "ai_abstract_generated": counts["ai_abstract"],
                "keywords_from_source": counts["source_keyword"],
                "keywords_extracted_from_pdf": counts["extracted_keyword"],
                "keywords_missing": counts["missing_keyword"],
            },
            "warnings": [
                *state.get("warnings", []),
                *warnings,
            ],
            "error": None,
        }

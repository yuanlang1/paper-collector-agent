import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import (
    AbstractEnrichNode,
    PdfFrontPageEnrichmentResult,
)


RUN_ID = "abstract-enrich-test"


class RecordingModel:
    def __init__(
        self,
        result: PdfFrontPageEnrichmentResult | Exception,
    ) -> None:
        self.result = result
        self.calls: list[list] = []

    async def ainvoke(self, messages: list):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class AbstractEnrichNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _run_node(
        self,
        *,
        model: RecordingModel,
        paper_info: dict,
        extract_side_effect,
    ) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            pdf_path = store.base_dir / RUN_ID / "paper.pdf"
            artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="download_pdfs",
                source="test",
                kind="paper_info_pdf_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_info": paper_info,
                            "pdf_download": {
                                "local_pdf_path": str(pdf_path),
                            },
                        }
                    ]
                },
            )
            node = AbstractEnrichNode(artifact_store=store, model=model)
            patch_kwargs = (
                {"side_effect": extract_side_effect}
                if isinstance(extract_side_effect, Exception)
                else {"return_value": extract_side_effect}
            )

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.abstract_enrich."
                "_extract_initial_pdf_text",
                **patch_kwargs,
            ):
                update = await node(
                    {
                        "run_id": RUN_ID,
                        "pdf_manifest_artifact_ref": artifact.artifact_uri,
                    }
                )

            manifest = await store.read_json_uri(
                update["abstract_manifest_artifact_ref"]
            )
            return update, manifest

    async def test_uses_one_pdf_front_page_request_for_all_fields(self) -> None:
        front_pages = "Abstract: PDF abstract\nKeywords: PDF keyword"
        model = RecordingModel(
            PdfFrontPageEnrichmentResult(
                paper_abstract="PDF abstract",
                keywords=["PDF keyword", "Second keyword"],
                ai_abstract="中文 AI 摘要",
            )
        )

        update, manifest = await self._run_node(
            model=model,
            paper_info={
                "paper_abstract": "source abstract",
                "keywords": "source keyword",
            },
            extract_side_effect=(front_pages, 4),
        )

        paper_info = manifest["papers"][0]["paper_info"]
        self.assertEqual(paper_info["paper_abstract"], "PDF abstract")
        self.assertEqual(paper_info["keywords"], "PDF keyword, Second keyword")
        self.assertEqual(paper_info["ai_abstract"], "中文 AI 摘要")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            json.loads(model.calls[0][1].content),
            {"pdf_front_pages": front_pages},
        )
        self.assertEqual(update["stage"], "recommending")

    async def test_keeps_source_fields_when_pdf_text_is_unavailable(self) -> None:
        model = RecordingModel(
            PdfFrontPageEnrichmentResult(
                paper_abstract="unused",
                keywords=["unused"],
                ai_abstract="unused",
            )
        )

        update, manifest = await self._run_node(
            model=model,
            paper_info={
                "paper_abstract": "source abstract",
                "keywords": "source keyword",
            },
            extract_side_effect=ValueError("no PDF text"),
        )

        paper = manifest["papers"][0]
        self.assertEqual(paper["paper_info"]["paper_abstract"], "source abstract")
        self.assertEqual(paper["paper_info"]["keywords"], "source keyword")
        self.assertIsNone(paper["paper_info"]["ai_abstract"])
        self.assertEqual(model.calls, [])
        self.assertEqual(update["status"], "partial_failed")

    async def test_keeps_source_fields_when_single_request_fails(self) -> None:
        model = RecordingModel(RuntimeError("LLM unavailable"))

        update, manifest = await self._run_node(
            model=model,
            paper_info={
                "paper_abstract": "source abstract",
                "keywords": "source keyword",
            },
            extract_side_effect=("PDF front pages", 2),
        )

        paper = manifest["papers"][0]
        self.assertEqual(paper["paper_info"]["paper_abstract"], "source abstract")
        self.assertEqual(paper["paper_info"]["keywords"], "source keyword")
        self.assertIsNone(paper["paper_info"]["ai_abstract"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(update["status"], "partial_failed")


if __name__ == "__main__":
    unittest.main()

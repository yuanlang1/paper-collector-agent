import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import (
    AbstractEnrichNode,
    PdfFrontPageEnrichmentResult,
)
from tests.chat_model_stub import as_chat_model


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
            node = AbstractEnrichNode(artifact_store=store, model=as_chat_model(model))
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
                        "child_run_id": f"child-{RUN_ID}",
                        "pdf_manifest_artifact_ref": artifact.artifact_uri,
                    }
                )

            manifest = await store.read_json_uri(
                update["abstract_manifest_artifact_ref"]
            )
            return update, manifest

    async def test_preserves_source_abstract_and_uses_one_request(self) -> None:
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
        self.assertEqual(paper_info["paper_abstract"], "source abstract")
        self.assertEqual(paper_info["keywords"], "PDF keyword, Second keyword")
        self.assertEqual(paper_info["ai_abstract"], "中文 AI 摘要")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            json.loads(next(message.content for message in model.calls[0] if isinstance(message, HumanMessage))),
            {"pdf_front_pages": front_pages, "source_abstract": "source abstract", "abstract_source": None},
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
        self.assertEqual(paper["paper_info"]["ai_abstract"], "unused")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(json.loads(next(message.content for message in model.calls[0] if isinstance(message, HumanMessage)))["pdf_front_pages"], "")
        self.assertEqual(paper["abstract_resolution"]["ai_summary_source"], "source_metadata")
        self.assertEqual(update["status"], "running")

    async def test_missing_source_abstract_generates_both_summaries(self) -> None:
        for generated in (False, True):
            with self.subTest(generated=generated):
                model = RecordingModel(PdfFrontPageEnrichmentResult(
                    paper_abstract="Original-language summary",
                    paper_abstract_generated=generated,
                    ai_abstract="整理后的中文摘要",
                ))
                _, manifest = await self._run_node(
                    model=model,
                    paper_info={"paper_abstract": "  "},
                    extract_side_effect=("Paper text", 2),
                )
                paper = manifest["papers"][0]
                self.assertEqual(paper["paper_info"]["paper_abstract"], "Original-language summary")
                self.assertEqual(paper["paper_info"]["ai_abstract"], "整理后的中文摘要")
                self.assertEqual(paper["abstract_resolution"]["source"],
                                 "generated_from_pdf" if generated else "pdf_front_pages")
                self.assertEqual(len(model.calls), 1)

    async def test_pdf_abstract_replaces_search_snippet(self) -> None:
        _, manifest = await self._run_node(
            model=RecordingModel(PdfFrontPageEnrichmentResult(
                paper_abstract="Complete PDF abstract", ai_abstract="中文摘要",
            )),
            paper_info={"paper_abstract": "Fragment …", "abstract_source": "search_snippet"},
            extract_side_effect=("Abstract: Complete PDF abstract", 1),
        )
        self.assertEqual(manifest["papers"][0]["paper_info"]["paper_abstract"], "Complete PDF abstract")

    async def test_without_source_or_pdf_does_not_invent_abstracts(self) -> None:
        model = RecordingModel(RuntimeError("must not be called"))
        _, manifest = await self._run_node(
            model=model, paper_info={"paper_abstract": None},
            extract_side_effect=ValueError("no PDF text"),
        )
        self.assertEqual(model.calls, [])
        self.assertIsNone(manifest["papers"][0]["paper_info"]["paper_abstract"])
        self.assertIsNone(manifest["papers"][0]["paper_info"]["ai_abstract"])

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
        self.assertEqual(update["status"], "running")

    async def test_empty_pdf_manifest_skips_pdf_and_model_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="download_pdfs",
                source="test",
                kind="paper_info_pdf_manifest_json",
                payload={"papers": []},
            )
            model = RecordingModel(
                PdfFrontPageEnrichmentResult(
                    paper_abstract="unused",
                    keywords=["unused"],
                    ai_abstract="未使用",
                )
            )
            node = AbstractEnrichNode(artifact_store=store, model=as_chat_model(model))

            with patch(
                "app.llm.graph.workflows.paper_search.nodes.abstract_enrich."
                "_extract_initial_pdf_text",
            ) as extract_pdf_text:
                update = await node(
                    {
                        "child_run_id": f"child-{RUN_ID}",
                        "pdf_manifest_artifact_ref": artifact.artifact_uri,
                    }
                )

            extract_pdf_text.assert_not_called()
            self.assertEqual(model.calls, [])
            self.assertEqual(update["stage"], "recommending")
            manifest = await store.read_json_uri(
                update["abstract_manifest_artifact_ref"]
            )
            self.assertEqual(manifest["papers"], [])

    async def test_limits_concurrency_and_keeps_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            papers = [
                {
                    "paper_info": {"title": f"Paper {index}"},
                    "pdf_download": {
                        "local_pdf_path": str(
                            store.base_dir / RUN_ID / f"paper-{index}.pdf"
                        )
                    },
                }
                for index in range(4)
            ]
            artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="download_pdfs",
                source="test",
                kind="paper_info_pdf_manifest_json",
                payload={"papers": papers},
            )
            started = asyncio.Event()
            release = asyncio.Event()

            class ConcurrentModel:
                def __init__(self) -> None:
                    self.active = 0
                    self.max_active = 0

                async def ainvoke(self, _messages: list):
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    if self.active == 3:
                        started.set()
                    try:
                        await release.wait()
                    finally:
                        self.active -= 1
                    return PdfFrontPageEnrichmentResult(
                        paper_abstract="PDF abstract",
                        keywords=["PDF keyword"],
                        ai_abstract="中文 AI 摘要",
                    )

            model = ConcurrentModel()
            with patch(
                "app.llm.graph.workflows.paper_search.nodes.abstract_enrich."
                "_extract_initial_pdf_text",
                return_value=("PDF front pages", 1),
            ):
                task = asyncio.create_task(
                    AbstractEnrichNode(artifact_store=store, model=as_chat_model(model))(
                        {
                            "child_run_id": f"child-{RUN_ID}",
                            "pdf_manifest_artifact_ref": artifact.artifact_uri,
                        }
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                self.assertEqual(model.max_active, 3)
                release.set()
                update = await task

            manifest = await store.read_json_uri(
                update["abstract_manifest_artifact_ref"]
            )
            self.assertEqual(
                [paper["paper_info"]["title"] for paper in manifest["papers"]],
                [f"Paper {index}" for index in range(4)],
            )

    async def test_cancellation_waits_for_in_flight_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            artifact = await store.write_json(
                run_id=RUN_ID,
                step_key="download_pdfs",
                source="test",
                kind="paper_info_pdf_manifest_json",
                payload={
                    "papers": [
                        {
                            "paper_info": {"title": "Paper"},
                            "pdf_download": {
                                "local_pdf_path": str(
                                    store.base_dir / RUN_ID / "paper.pdf"
                                )
                            },
                        }
                    ]
                },
            )
            started = asyncio.Event()

            class BlockingModel:
                def __init__(self) -> None:
                    self.cancelled = False

                async def ainvoke(self, _messages: list):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise

            model = BlockingModel()
            with patch(
                "app.llm.graph.workflows.paper_search.nodes.abstract_enrich."
                "_extract_initial_pdf_text",
                return_value=("PDF front pages", 1),
            ):
                task = asyncio.create_task(
                    AbstractEnrichNode(artifact_store=store, model=as_chat_model(model))(
                        {
                            "child_run_id": f"child-{RUN_ID}",
                            "pdf_manifest_artifact_ref": artifact.artifact_uri,
                        }
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            self.assertTrue(model.cancelled)


if __name__ == "__main__":
    unittest.main()

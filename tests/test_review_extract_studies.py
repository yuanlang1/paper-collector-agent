import asyncio
import json
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.infrastructure.grpc.paper_service_grpc_client import PaperServiceGrpcClient
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.review_generate.nodes.extract_studies import (
    ArticleProfile,
    ExtractStudiesNode,
)
from app.llm.graph.workflows.review_generate.nodes.finalize import finalize_task_review_node
from app.llm.graph.workflows.review_generate.nodes.load import LoadTaskCorpusNode
from app.protos.paper.v1 import paper_pb2


def _document(chunk_id: str, index: int, page: int, section_path: str, text: str,) -> Document:
    return Document(
        page_content=text,
        metadata={
            "paper_id": "1",
            "chunk_id": chunk_id,
            "chunk_index": index,
            "page_numbers": [page],
            "section_path": section_path,
        },
    )


class _ReadingIndex:
    def __init__(self):
        self.records = {}

    async def lookup(self, paper_id):
        return self.records.get(paper_id)

    async def save(self, payload):
        self.records[payload["paper_id"]] = payload


class _CorpusReader:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    async def read(self, paper_ids: list[str]) -> list[Document]:
        self.paper_ids = paper_ids
        return self.documents


class _Retrieval:
    def __init__(self, responses: dict[str, list[Document]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int, list[str]]] = []

    async def search(self, query: str, *, top_k: int, paper_ids: list[str],) -> list[Document]:
        self.calls.append((query, top_k, paper_ids))
        return self.responses.get(query, [])


class _Model:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.messages = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


class _FailingModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        raise RuntimeError("model unavailable")


def _tool_call(name: str, arguments: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": call_id}],)


class _PaperClient:
    async def get_task_review_papers(self, task_id: int) -> dict:
        return {
            "ok": True,
            "result": {
                "task_id": task_id,
                "task_status": "rag_completed",
                "papers": [
                    {
                        "paper_id": 1,
                        "title": "Paper",
                        "authors": [],
                        "paper_abstract": "Abstract",
                        "rag_status": "ready",
                        "chunk_count": 1,
                    }
                ],
            },
            "error": None,
        }


class _ReviewPapersStub:
    async def GetTaskReviewPapers(self, request, *, timeout):
        self.task_id = request.task_id
        self.timeout = timeout
        return paper_pb2.GetTaskReviewPapersResponse(
            success=True,
            task_id=request.task_id,
            task_status="RAG_COMPLETED",
            papers=[
                paper_pb2.TaskReviewPaperItem(
                    paper_id=1,
                    title="Paper",
                    paper_abstract=" Abstract ",
                    rag_status="READY",
                    chunk_count=1,
                )
            ],
        )


class _PaperServiceClient(PaperServiceGrpcClient):
    def __init__(self, stub: _ReviewPapersStub) -> None:
        self.stub = stub

    async def _get_stub(self) -> _ReviewPapersStub:
        return self.stub


class ExtractStudiesNodeTests(unittest.IsolatedAsyncioTestCase):
    documents = [
        _document("chunk-1", 0, 1, "Abstract", "The study addresses issue A."),
        _document("chunk-2", 1, 2, "Methods", "We use method B."),
        _document("chunk-3", 2, 3, "Introduction", "Background."),
        _document("chunk-4", 3, 4, "Discussion", "Finding C is discussed."),
    ]

    async def asyncSetUp(self) -> None:
        self._event_context = bind_event_context(
            EventContext(bus=EventBus(), run_id="review-test"),
        )
        self._event_context.__enter__()

    async def asyncTearDown(self) -> None:
        self._event_context.__exit__(None, None, None)

    @staticmethod
    def _state(corpus_artifact_ref: str) -> dict:
        return {
            "stage": "extracting_studies",
            "run_id": "review-test",
            "task_id": 1,
            "topic": "Topic",
            "language": "en",
            "paper_ids_snapshot": ["1"],
            "corpus_artifact_ref": corpus_artifact_ref,
            "warnings": [],
        }

    async def _corpus_artifact(
        self, store: LocalArtifactStore, papers: list[dict] | None = None,
    ):
        return await store.write_json(
            run_id="review-test",
            step_key="load_task_corpus",
            source="test",
            kind="corpus",
            payload={"papers": papers or [{"paper_id": 1, "title": "Paper"}],},
        )

    async def test_review_papers_rpc_maps_response(self) -> None:
        response = await _PaperServiceClient(_ReviewPapersStub()).get_task_review_papers(1)

        self.assertTrue(response["ok"])
        paper = response["result"]["papers"][0]
        self.assertEqual(paper["paper_abstract"], "Abstract")
        self.assertEqual(paper["rag_status"], "ready")

    async def test_load_corpus_preserves_paper_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            update = await LoadTaskCorpusNode(client=_PaperClient(), artifact_store=store,)(
                {"stage": "loading_corpus", "run_id": "review-test", "task_id": 1, "warnings": [],}
            )

            payload = await store.read_json_uri(update["corpus_artifact_ref"])
            self.assertEqual(update["stage"], "extracting_studies")
            self.assertEqual(payload["papers"][0]["abstract"], "Abstract")

    async def test_profile_loop_uses_opening_pages_then_current_paper_rag(self) -> None:
        model = _Model(
            [
                _tool_call("search_current_paper", {"query": "discussion finding C"}, "search-1",),
                _tool_call(
                    "submit_article_profile",
                    {
                        "core_problem": {
                            "summary": "Addresses issue A.",
                            "source_chunk_ids": ["chunk-1"],
                        },
                        "methods": {"summary": "Uses method B.", "source_chunk_ids": ["chunk-2"],},
                        "main_discussion": {
                            "summary": "Discusses finding C.",
                            "source_chunk_ids": ["chunk-4"],
                        },
                    },
                    "submit-1",
                ),
            ]
        )
        retrieval = _Retrieval({"discussion finding C": [self.documents[3]],})

        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=model,
                corpus_reader=_CorpusReader(self.documents),
                content_retrieval=retrieval,
                max_rounds=2,
            )(self._state(corpus.artifact_uri))

            payload = await store.read_json_uri(update["study_records_artifact_ref"])

        self.assertEqual(update["stage"], "generating_framework")
        self.assertEqual(payload["studies"][0]["core_problem"], "Addresses issue A.")
        self.assertEqual(payload["studies"][0]["methods"], "Uses method B.")
        self.assertEqual(payload["studies"][0]["main_discussion"], "Discusses finding C.")
        self.assertEqual(
            [call[2] for call in retrieval.calls], [["1"]],
        )
        first_prompt = model.messages[0][1].content
        self.assertIn("chunk-1", first_prompt)
        self.assertIn("chunk-3", first_prompt)
        self.assertIn("chunk-4", first_prompt)
        self.assertNotIn("topic", json.loads(first_prompt))
        self.assertIn("zh-CN", first_prompt)
        self.assertTrue(any(isinstance(message, ToolMessage) for message in model.messages[1]))

    async def test_all_not_reported_is_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model(
                    [
                        _tool_call(
                            "submit_article_profile",
                            {
                                field: {"summary": "not_reported", "source_chunk_ids": [],}
                                for field in ("core_problem", "methods", "main_discussion",)
                            },
                            "submit-1",
                        )
                    ]
                ),
                corpus_reader=_CorpusReader(self.documents),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))
            self.assertEqual(update["stage"], "failed")
            self.assertNotIn("study_records_artifact_ref", update)

    async def test_profile_rejects_unseen_chunk_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model(
                    [
                        _tool_call(
                            "submit_article_profile",
                            {
                                "core_problem": {
                                    "summary": "Addresses issue A.",
                                    "source_chunk_ids": ["unknown-chunk"],
                                },
                                "methods": {"summary": "not_reported", "source_chunk_ids": [],},
                                "main_discussion": {
                                    "summary": "not_reported",
                                    "source_chunk_ids": [],
                                },
                            },
                            "submit-1",
                        )
                    ]
                ),
                corpus_reader=_CorpusReader(self.documents),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))

            report = await store.read_json_uri(update["study_extraction_report_artifact_ref"])

        self.assertEqual(update["stage"], "failed")
        self.assertIn("unseen chunks", report["failures"][0]["error"])

    async def test_profiles_run_with_bounded_paper_concurrency(self) -> None:
        active = 0
        peak = 0

        async def extract_profile(*, paper, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                paper_id = str(paper["paper_id"])
                return ArticleProfile(
                    core_problem=f"problem {paper_id}",
                    methods=f"methods {paper_id}",
                    main_discussion=f"discussion {paper_id}",
                )
            finally:
                active -= 1

        papers = [{"paper_id": paper_id, "title": f"Paper {paper_id}"} for paper_id in range(1, 5)]
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, papers)
            node = ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model([]),
                corpus_reader=_CorpusReader(self.documents),
                content_retrieval=_Retrieval({}),
                max_concurrent_papers=2,
            )
            node._extract_profile = extract_profile
            update = await node(
                {
                    **self._state(corpus.artifact_uri),
                    "paper_ids_snapshot": [str(paper_id) for paper_id in range(1, 5)],
                }
            )
            payload = await store.read_json_uri(update["study_records_artifact_ref"])

        self.assertEqual(peak, 2)
        self.assertEqual(
            [study["paper_id"] for study in payload["studies"]], ["1", "2", "3", "4"],
        )

    def test_opening_documents_use_first_three_actual_page_numbers(self) -> None:
        documents = [
            _document("chunk-a", 0, 5, "A", "A"),
            _document("chunk-b", 1, 7, "B", "B"),
            _document("chunk-c", 2, 9, "C", "C"),
            _document("chunk-d", 3, 11, "D", "D"),
        ]

        opening = ExtractStudiesNode._opening_documents(documents)

        self.assertEqual(
            [item.metadata["chunk_id"] for item in opening], ["chunk-a", "chunk-b", "chunk-c"]
        )

    async def test_model_failure_blocks_review_and_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_FailingModel(),
                corpus_reader=_CorpusReader(self.documents),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))

            self.assertEqual(update["stage"], "failed")
            report_ref = update["study_extraction_report_artifact_ref"]
            report = await store.read_json_uri(report_ref)
            self.assertEqual(report["extracted_paper_ids"], [])
            self.assertEqual(report["failures"][0]["paper_id"], "1")

            result = await finalize_task_review_node(
                {
                    **update,
                    "active_tool_call": {"id": "call-1", "name": "task_review"},
                    "task_id": 1,
                }
            )
            self.assertIn(report_ref, result["last_action_result"]["artifact_refs"])


if __name__ == "__main__":
    unittest.main()

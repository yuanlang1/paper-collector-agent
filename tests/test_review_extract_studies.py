import asyncio
import json
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from app.infrastructure.oss import MarkdownObject
from app.events.bus import EventBus
from app.events.context import EventContext, bind_event_context
from app.events.adapter import sse_name_for
from app.infrastructure.grpc.paper_service_grpc_client import PaperServiceGrpcClient
from app.llm.artifacts.store import LocalArtifactStore
from app.llm.graph.workflows.progress import instrument_subagent_progress_node
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


PDF_SHA256 = "a" * 64
MARKDOWN = "# Abstract\nOSS abstract.\n# Methods\nOSS methods.\n# Conclusions\nOSS conclusion.\n"


class _ObjectStore:
    def __init__(self, markdown=MARKDOWN, error=None):
        self.markdown = markdown
        self.error = error
        self.calls = []

    async def get_markdown(self, *, pdf_sha256):
        self.calls.append(pdf_sha256)
        if self.error:
            raise self.error
        if self.markdown is None:
            return None
        return MarkdownObject(self.markdown, "md/paper.md", "checksum")


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
        self.messages.append(list(messages))
        return self.responses.pop(0)


class _FailingModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        raise RuntimeError("model unavailable")


def _tool_call(name: str, arguments: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": call_id}],)


def _profile_response(
    *,
    core_problem: str | None = None,
    methods: str | None = None,
    main_discussion: str | None = None,
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "core_problem": core_problem or "Addresses issue A.",
                "methods": methods or "Uses method B.",
                "main_discussion": main_discussion or "Discusses finding C.",
            }
        )
    )


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
            "child_run_id": "child-review-test",
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
            payload={"papers": papers or [{"paper_id": 1, "title": "Paper", "oss_name": PDF_SHA256}],},
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
                {"stage": "loading_corpus", "run_id": "review-test", "child_run_id": "child-review-test", "task_id": 1, "warnings": [],}
            )

            payload = await store.read_json_uri(update["corpus_artifact_ref"])
            self.assertEqual(update["stage"], "extracting_studies")
            self.assertEqual(payload["papers"][0]["abstract"], "Abstract")

    async def test_profile_loop_uses_oss_sections_then_current_paper_rag(self) -> None:
        model = _Model(
            [
                _tool_call(
                    "search_paper_index",
                    {"paper_id": "1", "querys": ["discussion finding C"]},
                    "search-1",
                ),
                _profile_response(),
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
                object_store=_ObjectStore(),
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
        initial = json.loads(first_prompt)
        self.assertNotIn("article_structure", initial)
        self.assertEqual(initial["initial_evidence"], ["# Abstract\nOSS abstract.", "# Methods\nOSS methods."])
        self.assertNotIn("OSS conclusion.", first_prompt)
        self.assertNotIn("chunk_id", first_prompt)
        self.assertNotIn("page_numbers", first_prompt)
        self.assertNotIn("topic", json.loads(first_prompt))
        self.assertIn("zh-CN", first_prompt)
        self.assertTrue(any(isinstance(message, ToolMessage) for message in model.messages[1]))
        self.assertEqual(model.tools[0]["function"]["name"], "search_paper_index")

    async def test_all_not_reported_is_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model(
                    [
                        _profile_response(
                            core_problem="not_reported",
                            methods="not_reported",
                            main_discussion="not_reported",
                        ),
                    ]
                ),
                object_store=_ObjectStore(),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))
            self.assertEqual(update["stage"], "failed")
            self.assertNotIn("study_records_artifact_ref", update)

    async def test_profile_accepts_three_string_fields(self) -> None:
        model = _Model([_profile_response()])
        reading_index = _ReadingIndex()
        object_store = _ObjectStore()
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, [{"paper_id": 1, "oss_name": PDF_SHA256.upper()}])
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=reading_index,
                model=model,
                object_store=object_store,
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))

            payload = await store.read_json_uri(update["study_records_artifact_ref"])

        self.assertEqual(update["stage"], "generating_framework")
        self.assertEqual(len(model.messages), 1)
        self.assertEqual(object_store.calls, [PDF_SHA256.upper()])
        self.assertEqual(payload["studies"][0]["methods"], "Uses method B.")
        self.assertEqual(reading_index.records["1"]["profile"], {
            "core_problem": "Addresses issue A.",
            "methods": "Uses method B.",
            "main_discussion": "Discusses finding C.",
        })

    async def test_profile_corrects_invalid_json_once(self) -> None:
        model = _Model([AIMessage(content="{invalid"), _profile_response()])
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=model,
                object_store=_ObjectStore(),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))

        self.assertEqual(update["stage"], "generating_framework")
        self.assertIn("invalid JSON", model.messages[1][-1].content)

    async def test_profile_accepts_multiple_search_calls_in_one_round(self) -> None:
        model = _Model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_paper_index",
                            "args": {"paper_id": "1", "querys": ["methods", "design"]},
                            "id": "search-1",
                        },
                        {
                            "name": "search_paper_index",
                            "args": {"paper_id": "1", "querys": ["discussion", "limitations"]},
                            "id": "search-2",
                        },
                    ],
                ),
                _profile_response(),
            ]
        )
        retrieval = _Retrieval({
            "methods": [self.documents[1]],
            "discussion": [self.documents[3]],
        })
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=model,
                object_store=_ObjectStore(),
                content_retrieval=retrieval,
                max_rounds=2,
            )(self._state(corpus.artifact_uri))

        self.assertEqual(update["stage"], "generating_framework")
        self.assertEqual([call[0] for call in retrieval.calls], ["methods", "design", "discussion", "limitations"])

    async def test_profile_tool_uses_current_paper_id(self) -> None:
        model = _Model(
            [
                _tool_call(
                    "search_paper_index",
                    {"paper_id": "2", "querys": ["methods"]},
                    "search-1",
                ),
                _profile_response(),
            ]
        )
        retrieval = _Retrieval({})
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=model,
                object_store=_ObjectStore(),
                content_retrieval=retrieval,
            )(self._state(corpus.artifact_uri))

        self.assertEqual(update["stage"], "generating_framework")
        self.assertEqual(retrieval.calls, [("methods", 4, ["1"])])

    async def test_profile_stops_at_model_loop_limit(self) -> None:
        model = _Model(
            [
                _tool_call(
                    "search_paper_index",
                    {"paper_id": "1", "querys": ["methods"]},
                    "search-1",
                ),
                _tool_call(
                    "search_paper_index",
                    {"paper_id": "1", "querys": ["discussion"]},
                    "search-2",
                ),
                _profile_response(),
            ]
        )
        retrieval = _Retrieval({"methods": [self.documents[1]]})
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=model,
                object_store=_ObjectStore(),
                content_retrieval=retrieval,
                max_rounds=2,
            )(self._state(corpus.artifact_uri))

        self.assertEqual(update["stage"], "failed")
        self.assertEqual(len(model.messages), 2)
        self.assertEqual(len(model.responses), 1)
        self.assertEqual([call[0] for call in retrieval.calls], ["methods"])
        self.assertEqual(update["error_code"], "PROFILE_PROTOCOL_ERROR")

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

        papers = [{"paper_id": paper_id, "title": f"Paper {paper_id}", "oss_name": PDF_SHA256} for paper_id in range(1, 5)]
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, papers)
            node = ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model([]),
                object_store=_ObjectStore(),
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

    async def test_multiple_papers_emit_only_reading_phase_events(self) -> None:
        async def extract_profile(*, paper, **_kwargs):
            paper_id = str(paper["paper_id"])
            return ArticleProfile(
                core_problem=f"problem {paper_id}",
                methods=f"methods {paper_id}",
                main_discussion=f"discussion {paper_id}",
            )

        papers = [
            {"paper_id": paper_id, "title": f"Paper {paper_id}", "oss_name": PDF_SHA256}
            for paper_id in range(1, 4)
        ]
        events = []
        bus = EventBus()

        async def collect(event) -> None:
            events.append(event)

        bus.subscribe(collect)
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, papers)
            node = ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_Model([]),
                object_store=_ObjectStore(),
                content_retrieval=_Retrieval({}),
            )
            node._extract_profile = extract_profile
            wrapped = instrument_subagent_progress_node(
                node_name="extract_studies",
                node=node,
                phase="reading",
                phase_label="提取论文研究信息",
            )
            with bind_event_context(EventContext(
                bus=bus,
                run_id="review-test",
                scope={"delegation_id": "review-1"},
            )):
                update = await wrapped(
                    {
                        **self._state(corpus.artifact_uri),
                        "paper_ids_snapshot": ["1", "2", "3"],
                    },
                    {},
                )

        self.assertEqual(
            [sse_name_for(event) for event in events],
            ["subagent_progress", "subagent_progress"],
        )
        self.assertEqual(
            [event.payload["data"]["phase_state"] for event in events],
            ["started", "completed"],
        )
        self.assertTrue(all(
            event.payload["delegation_id"] == "review-1"
            for event in events
        ))
        self.assertEqual(
            update["reading_summary"],
            {"total": 3, "succeeded": 3, "failed": 0},
        )

    async def test_model_failure_blocks_review_and_returns_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store)
            update = await ExtractStudiesNode(
                artifact_store=store,
                reading_index=_ReadingIndex(),
                model=_FailingModel(),
                object_store=_ObjectStore(),
                content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))

            self.assertEqual(update["stage"], "failed")
            self.assertEqual(
                update["reading_summary"],
                {"total": 1, "succeeded": 0, "failed": 1},
            )
            report_ref = update["study_extraction_report_artifact_ref"]
            report = await store.read_json_uri(report_ref)
            self.assertEqual(report["extracted_paper_ids"], [])
            self.assertEqual(report["failures"][0]["paper_id"], "1")

            result = await finalize_task_review_node(
                {
                    **update,
                    "active_tool_call": {"id": "call-1", "name": "task_review"},
                    "child_run_id": "child-review-test",
                    "task_id": 1,
                }
            )
            self.assertEqual(result["last_action_result"]["artifact_refs"], [])
            self.assertNotIn("artifact://", str(result["last_action_result"]))

    def test_markdown_context_takes_first_two_sections_without_title_selection(self) -> None:
        markdown = "# Introduction\nOpening.\n## Details\nDetails.\n# Abstract\nAbstract.\n# Conclusion\nConclusion."
        self.assertEqual(ExtractStudiesNode._markdown_context(markdown), [
            "# Introduction\nOpening.", "## Details\nDetails.",
        ])

    def test_markdown_context_ignores_headings_inside_code_fences(self) -> None:
        markdown = (
            "# 第一节\n正文。\n```markdown\n# Fake section\n```\n"
            "~~~\n###### Another fake section\n~~~\n# 第二节\n第二节正文。\n# 第三节\n略。"
        )
        sections = ExtractStudiesNode._markdown_context(markdown)
        self.assertEqual(len(sections), 2)
        self.assertTrue(sections[0].startswith("# 第一节"))
        self.assertIn("###### Another fake section", sections[0])
        self.assertEqual(sections[1], "# 第二节\n第二节正文。")

    def test_markdown_context_with_one_or_no_heading(self) -> None:
        for markdown in ("# Introduction\nOpening.", "Plain paper text."):
            with self.subTest(markdown=markdown):
                self.assertEqual(ExtractStudiesNode._markdown_context(markdown), [markdown])

    def test_markdown_context_limits_each_section_text(self) -> None:
        markdown = "# First\n" + "a" * 4000 + "\n# Second\n" + "b" * 4000 + "\n# Third\nExcluded."
        sections = ExtractStudiesNode._markdown_context(markdown)
        self.assertEqual([len(item) for item in sections], [3000, 3000])
        self.assertTrue(sections[0].startswith("# First"))
        self.assertTrue(sections[1].startswith("# Second"))

    async def test_oss_missing_empty_and_failure_stop_extraction(self) -> None:
        for markdown, oss_name, error, category, retryable in (
            (None, PDF_SHA256, None, "no_readable_content", False),
            ("  ", PDF_SHA256, None, "no_readable_content", False),
            (MARKDOWN, PDF_SHA256, RuntimeError("OSS unavailable"), "execution", True),
        ):
            with self.subTest(markdown=markdown, oss_name=oss_name, error=error):
                model = _Model([])
                retrieval = _Retrieval({})
                object_store = _ObjectStore(markdown, error)
                with tempfile.TemporaryDirectory() as directory:
                    store = LocalArtifactStore(base_dir=directory)
                    corpus = await self._corpus_artifact(store, [{"paper_id": 1, "oss_name": oss_name}])
                    update = await ExtractStudiesNode(
                        artifact_store=store, model=model, reading_index=_ReadingIndex(),
                        object_store=object_store, content_retrieval=retrieval,
                    )(self._state(corpus.artifact_uri))
                    report = await store.read_json_uri(update["study_extraction_report_artifact_ref"])
                self.assertEqual(update["stage"], "failed")
                self.assertEqual(report["failures"][0]["error_category"], category)
                self.assertEqual(update["retryable"], retryable)
                self.assertEqual(model.messages, [])
                self.assertEqual(retrieval.calls, [])
                self.assertEqual(object_store.calls, [oss_name])

    async def test_cached_profile_skips_oss_and_model(self) -> None:
        reading_index = _ReadingIndex()
        reading_index.records["1"] = {"paper_id": "1", "profile": {
            "core_problem": "Cached problem.", "methods": "Cached methods.",
            "main_discussion": "Cached discussion.",
        }}
        object_store = _ObjectStore(error=RuntimeError("must not read OSS"))
        model = _Model([])
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, [{"paper_id": 1}])
            update = await ExtractStudiesNode(
                artifact_store=store, reading_index=reading_index, model=model,
                object_store=object_store, content_retrieval=_Retrieval({}),
            )(self._state(corpus.artifact_uri))
            payload = await store.read_json_uri(update["study_records_artifact_ref"])
        self.assertEqual(update["stage"], "generating_framework")
        self.assertEqual(payload["studies"][0]["core_problem"], "Cached problem.")
        self.assertEqual(object_store.calls, [])
        self.assertEqual(model.messages, [])

    async def test_batch_failure_keeps_successful_profile_in_cache(self) -> None:
        reading_index = _ReadingIndex()
        model = _Model([_profile_response()])
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(base_dir=directory)
            corpus = await self._corpus_artifact(store, [
                {"paper_id": 1, "oss_name": PDF_SHA256}, {"paper_id": 2},
            ])
            update = await ExtractStudiesNode(
                artifact_store=store, reading_index=reading_index, model=model,
                object_store=_ObjectStore(), content_retrieval=_Retrieval({}),
            )({**self._state(corpus.artifact_uri), "paper_ids_snapshot": ["1", "2"]})
            report = await store.read_json_uri(update["study_extraction_report_artifact_ref"])
        self.assertEqual(update["stage"], "failed")
        self.assertEqual(update["reading_summary"], {"total": 2, "succeeded": 1, "failed": 1})
        self.assertEqual(report["extracted_paper_ids"], ["1"])
        self.assertIn("1", reading_index.records)
        self.assertNotIn("study_records_artifact_ref", update)


if __name__ == "__main__":
    unittest.main()

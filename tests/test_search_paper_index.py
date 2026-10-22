import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from pydantic import ValidationError

from app.llm.agent_build import build_tools
from app.llm.tools.search_tools.search_paper_index import (
    MAX_CHUNK_CHARS,
    MAX_RESULT_CHARS,
    SearchPaperIndexArgs,
    search_paper_index,
)


class _Retrieval:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def search(self, query, *, top_k, paper_ids):
        self.calls.append((query, top_k, paper_ids))
        return self.responses.get(query, [])


def _document(chunk_id, text):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "section_path": "Methods",
            "page_numbers": [2],
        },
    )


class SearchPaperIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_each_query_with_the_requested_paper_id(self):
        retrieval = _Retrieval({
            "methods": [_document("chunk-1", "Method text")],
            "results": [_document("chunk-2", "Result text")],
        })

        result = await search_paper_index(
            {"paper_id": "113", "querys": [" methods ", "results"]},
            content_retrieval=retrieval,
        )

        self.assertEqual(result["paper_id"], "113")
        self.assertEqual(result["results"][0]["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(result["results"][1]["chunks"][0]["page_numbers"], [2])
        self.assertEqual(
            retrieval.calls,
            [("methods", 4, ["113"]), ("results", 4, ["113"])],
        )

    async def test_empty_results_are_preserved(self):
        result = await search_paper_index(
            {"paper_id": "113", "querys": ["missing"]},
            content_retrieval=_Retrieval({}),
        )

        self.assertEqual(result["results"], [{"query": "missing", "chunks": []}])

    async def test_clips_chunk_and_total_result_lengths(self):
        retrieval = _Retrieval({
            "first": [_document("chunk-1", "a" * (MAX_CHUNK_CHARS + 1))],
            "second": [_document("chunk-2", "b" * MAX_RESULT_CHARS)],
        })

        result = await search_paper_index(
            {"paper_id": "113", "querys": ["first", "second"]},
            content_retrieval=retrieval,
        )

        chunks = [
            chunk
            for query_result in result["results"]
            for chunk in query_result["chunks"]
        ]
        self.assertLessEqual(max(len(chunk["text"]) for chunk in chunks), MAX_CHUNK_CHARS)
        self.assertLessEqual(sum(len(chunk["text"]) for chunk in chunks), MAX_RESULT_CHARS)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValidationError):
            SearchPaperIndexArgs.model_validate({"paper_id": "", "querys": []})
        with self.assertRaises(ValidationError):
            SearchPaperIndexArgs.model_validate({
                "paper_id": "113",
                "querys": ["query"] * 4,
            })
        with self.assertRaises(ValidationError):
            SearchPaperIndexArgs.model_validate({
                "paper_id": "113",
                "querys": ["x" * 161],
            })

    def test_is_not_registered_for_the_main_agent(self):
        self.assertNotIn("search_paper_index", {tool.name for tool in build_tools(None, user_id=1, model=Mock()).get_tools()})


if __name__ == "__main__":
    unittest.main()

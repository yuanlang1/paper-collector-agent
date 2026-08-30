import asyncio
import unittest

from app.llm.graph.workflows.paper_search.nodes.abstract_enrich import (
    AbstractEnrichNode,
    _extract_original_keywords,
    _normalize_keywords,
)
from app.llm.graph.workflows.paper_search.nodes.filter import (
    _to_paper_info_draft,
)


class PaperKeywordTests(unittest.TestCase):
    def test_generates_keywords_when_no_original_keywords_exist(self):
        class KeywordModel:
            async def ainvoke(self, _messages):
                return {
                    "keywords": [
                        "retrieval augmented generation",
                        "large language models",
                        "evaluation",
                    ]
                }

        node = AbstractEnrichNode(
            model=object(),
            keyword_model=KeywordModel(),
        )

        keywords = asyncio.run(
            node._generate_keywords(
                paper_info={
                    "title": "Evaluating RAG",
                    "paper_abstract": "A study of retrieval evaluation.",
                },
                initial_pdf_text=None,
            )
        )

        self.assertEqual(
            keywords,
            [
                "retrieval augmented generation",
                "large language models",
                "evaluation",
            ],
        )

    def test_extracts_keywords_from_pdf_front_pages(self):
        keywords = _extract_original_keywords(
            """Abstract
            This paper evaluates retrieval systems.
            Keywords: retrieval augmented generation; large language models,
            evaluation
            I. Introduction
            """
        )

        self.assertEqual(
            keywords,
            [
                "retrieval augmented generation",
                "large language models",
                "evaluation",
            ],
        )

    def test_normalization_deduplicates_keywords(self):
        keywords = _normalize_keywords(
            ["Retrieval", " retrieval ", "LLM；evaluation"]
        )

        self.assertEqual(keywords, ["Retrieval", "LLM", "evaluation"])

    def test_filter_does_not_copy_query_keywords_to_paper(self):
        paper_info = _to_paper_info_draft(
            paper={
                "title": "A Paper",
                "authors": ["Ada Lovelace"],
                "published": "2025-01-01",
            },
            source="arXiv",
        )

        self.assertIsNone(paper_info["keywords"])


if __name__ == "__main__":
    unittest.main()

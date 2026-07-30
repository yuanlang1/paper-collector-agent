import unittest

from app.llm.graph.workflows.paper_search.nodes.build_search_tag import (
    BuildSearchTagNode,
)


UNDERSTANDING = {
    "topic": "retrieval-augmented generation evaluation",
    "subfields": [],
    "intent": "benchmark",
    "yearFrom": 2023,
    "yearTo": 2026,
    "keywords": ["RAG evaluation"],
    "synonyms": [],
    "includeTerms": ["benchmark"],
    "excludeTerms": ["medical"],
    "requiresCode": None,
    "reasoning": "用户要求 RAG 评测基准论文。",
}


class FakeModel:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, _messages):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class BuildSearchTagNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedded_mode_does_not_emit_waiting_confirmation(
        self,
    ):
        node = BuildSearchTagNode(
            model=FakeModel(
                {
                    "yearTag": 0,
                    "paperTag": ["journal_article"],
                    "sourceTag": ["arXiv"],
                }
            ),
            skip_confirmation=True,
        )

        result = await node(
            {
                "original_prompt": "RAG evaluation",
                "query_understanding": UNDERSTANDING,
                "warnings": [],
            }
        )

        self.assertEqual(result["stage"], "creating_task")
        self.assertEqual(result["status"], "running")
        self.assertFalse(result["confirmation_required"])

    async def test_uses_model_tags_and_derives_year_tag(self):
        node = BuildSearchTagNode(
            model=FakeModel(
                {
                    "yearTag": 0,
                    "paperTag": ["proceedings_article"],
                    "sourceTag": ["arXiv", "DBLP"],
                }
            )
        )

        result = await node(
            {
                "original_prompt": "找 2023 到 2026 年 RAG benchmark 会议论文",
                "query_understanding": UNDERSTANDING,
                "warnings": [],
            }
        )

        self.assertEqual(result["search_tag"], {
            "yearTag": 4,
            "paperTag": [2],
            "sourceTag": ["arXiv", "DBLP"],
        })
        self.assertEqual(result["warnings"], [])

    async def test_falls_back_to_defaults_when_model_fails(self):
        node = BuildSearchTagNode(model=FakeModel(RuntimeError("model unavailable")))

        result = await node(
            {
                "original_prompt": "找 2023 到 2026 年 RAG benchmark 论文",
                "query_understanding": UNDERSTANDING,
                "warnings": [],
            }
        )

        self.assertEqual(result["search_tag"], {
            "yearTag": 4,
            "paperTag": [1, 2],
            "sourceTag": ["arXiv", "DBLP", "Google Scholar"],
        })
        self.assertIn("defaults were used", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()

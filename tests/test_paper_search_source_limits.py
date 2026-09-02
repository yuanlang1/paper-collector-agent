import unittest

from app.llm.agent import _source_query_plan_node
from app.services.setting_service import (
    source_limit_maxima,
    source_page_sizes,
    source_pagination_settings,
    validate_source_limits,
)


class SourceLimitsTests(unittest.IsolatedAsyncioTestCase):
    def test_pagination_is_derived_from_total_limits(self):
        maxima = source_limit_maxima()
        page_sizes = source_page_sizes()
        pagination = source_pagination_settings(maxima)

        self.assertEqual(pagination["arxiv_total_limit"], maxima["arxiv"])
        self.assertEqual(
            pagination["arxiv_max_pages"],
            maxima["arxiv"] // page_sizes["arxiv"],
        )
        self.assertEqual(pagination["dblp_total_limit"], maxima["dblp"])
        self.assertEqual(
            pagination["dblp_max_pages"],
            maxima["dblp"] // page_sizes["dblp"],
        )
        self.assertEqual(
            pagination["google_total_limit"],
            maxima["google_scholar"],
        )
        self.assertEqual(
            pagination["google_max_pages"],
            maxima["google_scholar"] // page_sizes["google_scholar"],
        )

    def test_source_limits_require_each_known_source(self):
        with self.assertRaises(ValueError):
            validate_source_limits({"arxiv": 1})

    async def test_query_plan_uses_run_source_limit_snapshot(self):
        class QueryPlanModel:
            async def ainvoke(self, _messages):
                return {
                    "plans": [
                        {
                            "source": "arXiv",
                            "display_query": "rag evaluation",
                            "reasoning": "topic keywords",
                            "arguments": {"query": "rag evaluation"},
                        },
                        {
                            "source": "DBLP",
                            "display_query": "rag evaluation",
                            "reasoning": "topic keywords",
                            "arguments": {"query": "rag evaluation"},
                        },
                        {
                            "source": "Google Scholar",
                            "display_query": "rag evaluation",
                            "reasoning": "topic keywords",
                            "arguments": {"query": "rag evaluation"},
                        },
                    ]
                }

        maxima = source_limit_maxima()
        result = await _source_query_plan_node(QueryPlanModel())(
            {
                "paper_search_source_limits": maxima,
                "query_understanding": {
                    "topic": "RAG evaluation",
                    "intent": "benchmark",
                    "reasoning": "user requested evaluation papers",
                },
                "search_tag": {
                    "yearTag": 0,
                    "paperTag": [1, 2],
                    "sourceTag": ["arXiv", "DBLP", "Google Scholar"],
                },
            }
        )
        plans = {plan["source"]: plan for plan in result["source_query_plans"]}

        self.assertEqual(
            plans["arXiv"]["arguments"]["total_limit"], maxima["arxiv"]
        )
        self.assertEqual(
            plans["DBLP"]["arguments"]["total_limit"], maxima["dblp"]
        )
        self.assertEqual(
            plans["Google Scholar"]["arguments"]["total_limit"],
            maxima["google_scholar"],
        )


if __name__ == "__main__":
    unittest.main()
